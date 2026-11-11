"""Pack chains with geometric checks before invoking Packmol."""

import json
import logging
import math
import os
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from rdkit import Chem
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from openff.toolkit import Molecule, Topology
from openff.units import unit

logger = logging.getLogger(__name__)


def _check_chain_geometry(mol, index):
    """Detect severe non-local atomic clashes; bonded and 1-3/1-4 pairs are excluded."""
    if mol is None or mol.GetNumConformers() == 0:
        raise ValueError(f"Chain {index} must have 3D coordinates")
    coords = np.asarray(mol.GetConformer().GetPositions(), dtype=float)
    if not np.isfinite(coords).all():
        raise ValueError(f"Chain {index} has non-finite coordinates")
    table = Chem.GetPeriodicTable()
    radii = [table.GetRvdw(a.GetAtomicNum()) for a in mol.GetAtoms()]
    if not radii:
        raise ValueError(f"Chain {index} has no atoms")
    for i, j in sorted(cKDTree(coords).query_pairs(max(radii))):
        # Atoms separated by at most three bonds have geometry governed by
        # local bonded terms; they must not be treated as chain self-clashes.
        visited = {i}
        frontier = {i}
        for _ in range(3):
            frontier = {n.GetIdx() for k in frontier for n in mol.GetAtomWithIdx(k).GetNeighbors()} - visited
            visited.update(frontier)
        if j in visited:
            continue
        distance = float(np.linalg.norm(coords[i] - coords[j]))
        threshold = 0.5 * (radii[i] + radii[j]) * 0.5
        if distance < threshold:
            raise ValueError(
                f"Chain {index} has an internal nonbonded overlap: atoms {i} and {j} "
                f"are {distance:.3f} Å apart (clash threshold {threshold:.3f} Å). "
                "Relax or regenerate this chain before packing."
            )
    # The diagonal bounds every interatomic distance even after rotation.
    return float(np.linalg.norm(np.ptp(coords, axis=0)))


def _run_packmol_attempt(molecules, density, tolerance, seed, timeout, disable_movebad):
    """Run one isolated Packmol attempt and terminate its process tree on timeout."""
    with tempfile.TemporaryDirectory() as directory:
        base = Path(directory)
        request_path = base / 'request.json'
        result_path = base / 'topology.json'
        workdir = base / 'packmol'
        request_path.write_text(json.dumps({
            'molecules': [molecule.to_json() for molecule in molecules],
            'density': density, 'tolerance': tolerance, 'seed': seed,
            'workdir': str(workdir), 'disable_movebad': disable_movebad,
        }))
        worker = Path(__file__).with_name('_packing_worker.py')
        process = subprocess.Popen(
            [sys.executable, str(worker), str(request_path), str(result_path)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            start_new_session=(os.name == 'posix'),
        )
        try:
            output, _ = process.communicate(timeout=timeout)
        except BaseException:
            if process.poll() is None:
                if os.name == 'posix':
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                else:
                    process.kill()
            process.communicate()
            raise
        if process.returncode or not result_path.is_file():
            detail = output.decode('utf-8', errors='replace')[-3000:]
            raise RuntimeError(f'Packmol worker failed: {detail}')
        return Topology.from_json(result_path.read_text())


class PackmolPacker:
    """Pack a set of chains at a geometry-adjusted initial density."""

    def __init__(self, density: float = 0.3, max_attempts: int = 5,
                 tolerance: float = 2.0, timeout: float = 60.0, seed: int = None):
        if isinstance(density, bool) or not isinstance(density, (int, float)) or not math.isfinite(density) or density <= 0:
            raise ValueError("density must be a positive finite number in g/mL")
        if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)) or not math.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("tolerance must be a positive finite number in Å")
        if isinstance(max_attempts, bool) or not isinstance(max_attempts, int) or max_attempts < 1:
            raise ValueError("max_attempts must be a positive integer")
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be a positive finite number of seconds")
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**31):
            raise ValueError("seed must be an integer between 0 and 2**31 - 1 or None")
        self.timeout = float(timeout)
        self.seed = seed
        self.density = density * unit.gram / unit.milliliter
        self.max_attempts = max_attempts
        self.tolerance = tolerance * unit.angstrom
        self.effective_density = density
        self.initial_pack_density = density

    def _rdkit_to_openff(self, mol):
        mol = Chem.Mol(mol)
        try:
            Chem.AssignStereochemistryFrom3D(mol, confId=0, force=True, cleanIt=True)
            Chem.AssignStereochemistry(mol, force=True, cleanIt=True, flagPossibleStereoCenters=True)
        except Exception:
            pass
        return Molecule.from_rdkit(mol, allow_undefined_stereo=True)

    def pack(self, chains):
        if not chains:
            raise ValueError("List of chains is empty")
        extents = [_check_chain_geometry(mol, i) for i, mol in enumerate(chains)]
        molecules = [self._rdkit_to_openff(mol) for mol in chains]
        # A cube wider than the diagonal of each axis-aligned bounding box
        # fits every chain in any orientation. Interchange reserves tolerance
        # at the box boundary; an extra Å provides a small numerical margin.
        side = max(extents) + self.tolerance.m_as(unit.angstrom) + 1.0
        total_mass = sum(atom.mass.m_as(unit.dalton) for mol in molecules for atom in mol.atoms)
        max_density = total_mass * 1.66053906660 / side**3
        requested = self.density.m_as(unit.gram / unit.milliliter)
        self.effective_density = min(requested, max_density)
        if self.effective_density < requested:
            logger.warning(
                "Requested initial density %.6g g/mL is too high for the chain dimensions; "
                "using %.6g g/mL (box side %.3f Å) before packing.",
                requested, self.effective_density, side,
            )
        # For large dense systems, give Packmol room to place the rigid chains,
        # then geometrically compress the periodic box to the requested density.
        # The target density is never silently reduced by this strategy.
        use_compression = (
            sum(mol.GetNumAtoms() for mol in chains) >= 1000
            and self.effective_density > 0.05
            and all(len(Chem.GetMolFrags(mol)) == 1 for mol in chains)
        )
        self.initial_pack_density = min(self.effective_density, 0.05) if use_compression else self.effective_density
        if use_compression:
            logger.info("Packing initially at %.6g g/mL, then compressing rigid chains to %.6g g/mL",
                        self.initial_pack_density, self.effective_density)
        rng = np.random.default_rng(self.seed)
        last_error = None
        # Keep the requested density as the first choice, then give Packmol
        # progressively more room. A bad target density must not prevent a
        # valid, lower-density periodic system from being returned.
        densities = [self.initial_pack_density]
        for _ in range(4):
            next_density = densities[-1] * 0.65
            if next_density < 1e-4:
                break
            densities.append(next_density)

        for density_index, density in enumerate(densities):
            attempts = self.max_attempts if density_index == 0 else min(2, self.max_attempts)
            for attempt in range(1, attempts + 1):
                attempt_seed = int(rng.integers(1, 2**31 - 1))
                logger.info(
                    "Packmol attempt %d/%d at %.6g g/mL: seed=%d, disable_movebad=%s",
                    attempt, attempts, density, attempt_seed, attempt % 2 == 0,
                )
                try:
                    topology = _run_packmol_attempt(
                        molecules, density,
                        self.tolerance.m_as(unit.angstrom), attempt_seed, self.timeout,
                        attempt % 2 == 0,
                    )
                    if use_compression:
                        from ._rigid_compression import compress_topology
                        topology = compress_topology(
                            topology, chains, density, self.effective_density,
                            self.tolerance.m_as(unit.angstrom), seed=attempt_seed,
                            timeout=self.timeout,
                        )
                    else:
                        self.effective_density = density
                    self.initial_pack_density = density
                    return topology
                except Exception as exc:
                    # Packmol, subprocess startup, and rigid-box compression
                    # failures all enter the same bounded fallback ladder.
                    last_error = exc
                    logger.warning("Packing attempt failed at %.6g g/mL: %s", density, exc)

        # Last resort: randomized periodic placement, then rigid-body push-off
        # over the complete box. If compression cannot reach the requested
        # density, keep the clash-free dilute periodic system.
        topology, safe_density = _random_periodic_pack(
            molecules, chains, extents, self.tolerance.m_as(unit.angstrom),
            seed=int(rng.integers(1, 2**31 - 1)),
        )
        if safe_density > self.effective_density:
            scale = (safe_density / self.effective_density) ** (1.0 / 3.0)
            topology, safe_density = _random_periodic_pack(
                molecules, chains, extents, self.tolerance.m_as(unit.angstrom),
                seed=int(rng.integers(1, 2**31 - 1)), scale=scale,
            )
            self.initial_pack_density = safe_density
            self.effective_density = safe_density
            return topology
        try:
            from ._rigid_compression import compress_topology
            pushed = compress_topology(
                topology, chains, safe_density, self.effective_density,
                self.tolerance.m_as(unit.angstrom),
                seed=int(rng.integers(1, 2**31 - 1)), timeout=max(self.timeout, 30.0),
            )
            self.initial_pack_density = self.effective_density
            return pushed
        except Exception as exc:
            logger.warning(
                "Whole-box push-off could not reach %.6g g/mL; returning safe "
                "random periodic packing at %.6g g/mL (last packing error: %s; push-off: %s)",
                self.effective_density, safe_density, last_error, exc,
            )
            self.initial_pack_density = safe_density
            self.effective_density = safe_density
            return topology


def pack_chains(chains, density=0.3, max_attempts=5, tolerance=2.0, timeout=60.0, seed=None):
    """Pack RDKit polymer chains and return an OpenFF topology."""
    return PackmolPacker(density=density, max_attempts=max_attempts, tolerance=tolerance,
                         timeout=timeout, seed=seed).pack(chains)


def _random_periodic_pack(molecules, chains, extents, tolerance, *, seed, scale=1.0):
    """Place rigid chains in randomized periodic lattice cells with guaranteed clearance."""
    from openff.toolkit import Topology

    count = len(chains)
    grid = int(math.ceil(count ** (1.0 / 3.0)))
    # Each chain fits inside a sphere whose radius is bounded by its box
    # diagonal. Separate cell centers by more than two radii plus tolerance.
    spacing = (2.0 * max(extents) + tolerance + 1.0) * scale
    side = grid * spacing
    rng = np.random.default_rng(seed)
    cells = np.array(
        [(i, j, k) for i in range(grid) for j in range(grid) for k in range(grid)],
        dtype=float,
    )[:count]
    cells = cells[rng.permutation(count)]
    blocks = []
    for molecule, cell in zip(chains, cells):
        coords = np.asarray(molecule.GetConformer().GetPositions(), dtype=float)
        centered = coords - coords.mean(axis=0)
        rotation = Rotation.random(random_state=rng).as_matrix()
        center = (cell + 0.5) * spacing
        blocks.append(centered @ rotation.T + center)
    positions = np.concatenate(blocks, axis=0)
    topology = Topology.from_molecules(molecules)
    topology.set_positions(positions * unit.angstrom)
    topology.box_vectors = np.eye(3) * side * unit.angstrom
    mass = sum(atom.mass.m_as(unit.dalton) for mol in molecules for atom in mol.atoms)
    density = mass * 1.66053906660 / side**3
    logger.warning(
        "Using randomized whole-box periodic fallback: density %.6g g/mL, box side %.3f Å",
        density, side,
    )
    return topology, density
