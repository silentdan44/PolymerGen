"""Resolve inter-chain contacts during gradual rigid-body box compression."""

import time

import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from openff.units import unit


def _inter_pairs(positions, molecule_index, side, cutoff):
    """Return inter-molecule pairs and minimum-image displacement vectors."""
    wrapped = positions % side
    pairs = cKDTree(wrapped, boxsize=side).query_pairs(cutoff, output_type='ndarray')
    if not len(pairs):
        return pairs, np.empty((0, 3))
    pairs = pairs[molecule_index[pairs[:, 0]] != molecule_index[pairs[:, 1]]]
    delta = wrapped[pairs[:, 0]] - wrapped[pairs[:, 1]]
    delta -= side * np.rint(delta / side)
    return pairs, delta


def _rigid_energy_gradient(x, local, base, molecule_index, side, cutoff):
    """Penalty and analytical gradient for periodic rigid-body contacts."""
    count = len(base)
    translations = x[:3 * count].reshape(count, 3)
    rotation_vectors = x[3 * count:].reshape(count, 3)
    matrices = Rotation.from_rotvec(rotation_vectors).as_matrix()
    rotated = np.einsum('nij,nj->ni', matrices[molecule_index], local)
    positions = rotated + base[molecule_index] + translations[molecule_index]
    pairs, delta = _inter_pairs(positions, molecule_index, side, cutoff)
    distances = np.linalg.norm(delta, axis=1).clip(min=1e-10)
    overlap = cutoff - distances
    energy = 0.5 * np.dot(overlap, overlap)
    force = -(overlap / distances)[:, None] * delta
    atom_grad = np.zeros_like(positions)
    np.add.at(atom_grad, pairs[:, 0], force)
    np.add.at(atom_grad, pairs[:, 1], -force)
    translation_grad = np.zeros((count, 3))
    np.add.at(translation_grad, molecule_index, atom_grad)
    torque = np.zeros((count, 3))
    np.add.at(torque, molecule_index, np.cross(rotated, atom_grad))
    theta = np.linalg.norm(rotation_vectors, axis=1)
    a = np.where(theta < 1e-7, 0.5, (1 - np.cos(theta)) / np.maximum(theta**2, 1e-30))
    b = np.where(theta < 1e-7, 1 / 6, (theta - np.sin(theta)) / np.maximum(theta**3, 1e-30))
    cross = np.cross(rotation_vectors, torque)
    rotation_grad = torque - a[:, None] * cross + b[:, None] * np.cross(rotation_vectors, cross)
    return float(energy), np.r_[translation_grad.ravel(), rotation_grad.ravel()]


def _unwrap_chain(coords, molecule, side):
    """Unwrap a connected chain using its covalent graph and minimum images."""
    adjacency = [[] for _ in range(len(coords))]
    for bond in molecule.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        adjacency[i].append(j)
        adjacency[j].append(i)
    unwrapped = np.empty_like(coords)
    unwrapped[0] = coords[0]
    seen = {0}
    queue = [0]
    for i in queue:
        for j in adjacency[i]:
            if j not in seen:
                delta = coords[j] - coords[i]
                delta -= side * np.rint(delta / side)
                unwrapped[j] = unwrapped[i] + delta
                seen.add(j)
                queue.append(j)
    if len(seen) != len(coords):
        raise ValueError('Rigid compression requires each molecule to be connected')
    return unwrapped


def compress_topology(topology, chains, start_density, target_density, tolerance,
                      *, seed=None, timeout=300.0):
    """Compress a packed cubic topology without changing internal chain geometry.

    The returned topology is modified in place. A failure never returns a
    partially compressed result. The operation is geometric, not MD equilibration.
    """
    if not 0 < start_density <= target_density:
        raise ValueError('Compression densities must be positive and increasing')
    if start_density == target_density:
        return topology
    vectors = np.asarray(topology.box_vectors.m_as(unit.angstrom), dtype=float)
    side = float(vectors[0, 0])
    if not np.allclose(vectors, np.eye(3) * side):
        raise ValueError('Rigid compression requires a cubic box')
    coords = np.asarray(topology.get_positions().m_as(unit.angstrom), dtype=float)
    counts = [mol.GetNumAtoms() for mol in chains]
    if len(coords) != sum(counts):
        raise ValueError('Topology atom count does not match input chains')
    molecule_index = np.repeat(np.arange(len(chains)), counts)
    blocks = np.split(coords, np.cumsum(counts)[:-1])
    unwrapped = [_unwrap_chain(block, mol, side) for block, mol in zip(blocks, chains)]
    base = np.array([block.mean(axis=0) for block in unwrapped]) % side
    local = np.concatenate([block - block.mean(axis=0) for block in unwrapped])
    count = len(chains)
    cutoff = tolerance + 0.015
    rng = np.random.default_rng(seed)
    deadline = time.monotonic() + timeout
    steps = max(1, int(np.ceil(np.log(target_density / start_density) / np.log(1.23))))
    for density in np.geomspace(start_density, target_density, steps + 1)[1:]:
        if time.monotonic() >= deadline:
            raise TimeoutError(f'Rigid compression timed out after {timeout:g} seconds')
        next_side = side * (start_density / density)**(1 / 3)
        base = (base % side) * next_side / side
        side = next_side
        start_density = density
        best = None
        x0 = np.zeros(6 * count)
        def objective(x):
            if time.monotonic() >= deadline:
                raise TimeoutError(f'Rigid compression timed out after {timeout:g} seconds')
            return _rigid_energy_gradient(x, local, base, molecule_index, side, cutoff)
        for retry in range(8):
            result = minimize(objective, x0, jac=True, method='L-BFGS-B',
                              options={'maxiter': 1500, 'ftol': 1e-13, 'gtol': 1e-6, 'maxls': 25})
            if best is None or result.fun < best.fun:
                best = result
            if best.fun < 1e-6:
                break
            x0 = best.x.copy()
            translations = x0[:3 * count].reshape(count, 3)
            rotations = x0[3 * count:].reshape(count, 3)
            matrices = Rotation.from_rotvec(rotations).as_matrix()
            rotated = np.einsum('nij,nj->ni', matrices[molecule_index], local)
            positions = rotated + base[molecule_index] + translations[molecule_index]
            pairs, delta = _inter_pairs(positions, molecule_index, side, cutoff)
            if len(pairs):
                weights = np.maximum(cutoff - np.linalg.norm(delta, axis=1), 0)**2
                score = np.bincount(np.r_[molecule_index[pairs[:, 0]], molecule_index[pairs[:, 1]]],
                                    weights=np.r_[weights, weights], minlength=count)
                candidates = np.argsort(score)[-min(count, max(2, min(8, retry + 2))):]
                chosen = int(rng.choice(candidates))
                translations[chosen] += rng.normal(size=3) * (1.0 + retry * 0.5)
                rotations[chosen] += rng.normal(size=3) * 0.3
        if best.fun >= 1e-6:
            raise RuntimeError(f'Could not eliminate inter-chain overlaps at density {density:.6g} g/mL')
        translations = best.x[:3 * count].reshape(count, 3)
        matrices = Rotation.from_rotvec(best.x[3 * count:].reshape(count, 3)).as_matrix()
        local = np.einsum('nij,nj->ni', matrices[molecule_index], local)
        base = (base + translations) % side
    final_positions = local + base[molecule_index]
    pairs, _ = _inter_pairs(final_positions, molecule_index, side, tolerance - 1e-5)
    if len(pairs):
        raise RuntimeError(f'Rigid compression left {len(pairs)} inter-chain overlaps')
    topology.set_positions(final_positions * unit.angstrom)
    topology.box_vectors = np.eye(3) * side * unit.angstrom
    return topology
