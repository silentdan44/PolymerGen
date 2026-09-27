"""
Polymerization controller.

The Polymerizer class orchestrates the step-by-step assembly of a polymer
chain from a list of Monomer objects, using retry and rollback logic to
recover from steric clashes.
"""

import time
from dataclasses import dataclass
import numpy as np
from . import utils
from . import poly
from .utils import logger
from .optimizers import BaseOptimizer, get_optimizer


@dataclass(frozen=True)
class BuildConfig:
    """Validated parameters for polymer chain growth (distances in Å)."""

    bond_length: float = 1.5
    dihedral: float = np.pi
    random_rot: bool = True
    dist_min: float = 1.8
    retry: int = 100
    rollback: int = 5
    retry_step: int = 200
    check_bond_length: bool = True

    def __post_init__(self):
        if self.bond_length <= 0 or self.dist_min < 0:
            raise ValueError('bond_length must be > 0 and dist_min must be >= 0')
        if self.retry < 0 or self.retry_step < 1 or self.rollback < 0:
            raise ValueError('retry and rollback must be >= 0 and retry_step must be >= 1')


class Polymerizer:
    """
    Controller for building a polymer chain from a list of monomers.

    The build process iteratively connects monomers using
    poly.connect_mols, optionally applying a structure optimizer
    (e.g. MMFF or OpenFF) after each connection, and validates the
    resulting 3D geometry. If a clash is detected, the step is retried
    with a new random rotation; after too many failures, the chain is
    rolled back several steps and the process restarts.

    Attributes:
        monomers: List of Monomer objects to be joined.
        n: Number of monomers (== len(monomers)).
        optimizer: Structure optimizer applied after each connection.
    """

    def __init__(self, monomers, bond_length=1.5, dihedral=np.pi, random_rot=True,
                 dist_min=1.8, retry=100, rollback=5, retry_step=200,
                 check_bond_length=True, optimizer=None, optimizer_options=None,
                 config=None, seed=None):
        """
        Initialize the Polymerizer.

        Args:
            monomers: List of Monomer objects.
            bond_length: Target length for new bonds (Angstrom).
            dihedral: Target dihedral angle around new bonds (radians).
            random_rot: If True, use a random dihedral at each step.
            dist_min: Minimum allowed nonbonded heavy-atom distance (Å).
            retry: Max number of additional full-chain attempts after the first.
            rollback: Number of steps to roll back on failure.
            retry_step: Max attempts per polymerization step.
            check_bond_length: If True, also validate bond lengths.
            optimizer: Structure optimizer. Can be:
                - None (no optimization)
                - A BaseOptimizer instance (e.g. MMFFOptimizer)
                - A string ('mmff' or 'openff')
                - A dict like {'name': 'mmff', 'max_iters': 100}
        """
        if config is not None:
            if not isinstance(config, BuildConfig):
                raise TypeError('config must be a BuildConfig instance')
            bond_length = config.bond_length
            dihedral = config.dihedral
            random_rot = config.random_rot
            dist_min = config.dist_min
            retry = config.retry
            rollback = config.rollback
            retry_step = config.retry_step
            check_bond_length = config.check_bond_length
        self.monomers = monomers
        self.n = len(monomers)
        self.bond_length = bond_length
        self.dihedral = dihedral
        self.random_rot = random_rot
        self.dist_min = dist_min
        self.retry = retry
        self.rollback = rollback
        self.retry_step = retry_step
        self.check_bond_length = check_bond_length
        self.seed = seed
        if optimizer_options is not None:
            if not isinstance(optimizer, str):
                raise ValueError('optimizer_options requires optimizer to be a name such as "mmff"')
            if isinstance(optimizer_options, dict):
                if 'name' in optimizer_options:
                    raise ValueError('optimizer_options must not contain a name key')
                optimizer = {'name': optimizer, **optimizer_options}
            else:
                raise TypeError('optimizer_options must be a dict')
        if not isinstance(retry, int) or not isinstance(retry_step, int) or not isinstance(rollback, int) or \
           retry < 0 or retry_step < 1 or rollback < 0:
            raise ValueError('retry and rollback must be >= 0 and retry_step must be >= 1')

        # Parse optimizer into a BaseOptimizer instance
        self.optimizer = self._parse_optimizer(optimizer)

        self._last_attempts = 0
        self._last_rollbacks = 0
        self._last_failure_reason = None

    @classmethod
    def from_smiles(cls, smiles, count, **kwargs):
        """Create a polymerizer from one monomer SMILES and a repeat count."""
        if not isinstance(count, int) or count < 1:
            raise ValueError('count must be a positive integer')
        from .monomer import Monomer
        seed = kwargs.get('seed')
        monomer = Monomer(smiles, seed=seed)
        return cls([monomer.copy() for _ in range(count)], **kwargs)

    def _parse_optimizer(self, optimizer):
        """
        Convert various optimizer specifications into a BaseOptimizer instance.

        Args:
            optimizer: None, a BaseOptimizer, a string name, or a dict.

        Returns:
            A BaseOptimizer instance, or None.

        Raises:
            ValueError: If the optimizer specification is invalid.
        """
        if optimizer is None:
            return None
        if isinstance(optimizer, BaseOptimizer):
            return optimizer
        if isinstance(optimizer, str):
            return get_optimizer(optimizer)
        if isinstance(optimizer, dict):
            opts = dict(optimizer)
            name = opts.pop('name')
            return get_optimizer(name, **opts)
        raise ValueError(f"Invalid optimizer type: {type(optimizer)}")

    def build_mol(self):
        """
        Run the polymerization process and return the final polymer Mol.

        Returns:
            RDKit Mol object of the polymer with 3D coordinates, or
            None if the build failed after exhausting all retries.
        """
        if self.n < 1:
            raise ValueError('Number of monomers must be >= 1')

        self._last_attempts = 0
        self._last_rollbacks = 0
        self._last_failure_reason = None
        rng = np.random.default_rng(self.seed)

        t0 = time.time()
        logger.info(f'Start polymerization: n={self.n}')
        if self.optimizer is not None:
            logger.info(f'Optimizer: {self.optimizer.name}')

        poly_mol = utils.deepcopy_mol(self.monomers[0].mol)
        poly_history = [utils.deepcopy_mol(poly_mol)]
        history_start_step = 0
        history_limit = self.rollback + 1

        total_attempts = 0
        total_rollbacks = 0
        retries_left = self.retry
        start_step = 0

        while start_step < self.n - 1:
            step = start_step
            success_chain = True

            while step < self.n - 1:
                piece = utils.deepcopy_mol(self.monomers[step + 1].mol)
                previous_atom_count = poly_mol.GetNumAtoms()
                removed_head_idx = piece.GetIntProp('head_idx')
                expected_atom_count = previous_atom_count + piece.GetNumAtoms() - 2
                success = False

                for attempt in range(self.retry_step):
                    total_attempts += 1
                    self._last_attempts += 1
                    poly_trial = poly.connect_mols(
                        poly_mol, piece,
                        bond_length=self.bond_length,
                        dihedral=self.dihedral,
                        random_rot=self.random_rot,
                        rng=rng,
                    )
                    if poly_trial is None or poly_trial.GetNumAtoms() != expected_atom_count:
                        logger.warning(
                            f'Molecule connection failed at step {step+1}, attempt {attempt+1}'
                        )
                        continue

                    # The old tail atom is removed first; then the incoming
                    # head atom is removed at its concatenated index.
                    incoming_offset = previous_atom_count - 1
                    added_indices = [
                        incoming_offset + atom_idx - (atom_idx > removed_head_idx)
                        for atom_idx in range(piece.GetNumAtoms())
                        if atom_idx != removed_head_idx
                    ]

                    # A force-field/parameterization failure invalidates this
                    # trial; let the normal retry/rollback path recover.
                    if self.optimizer is not None:
                        try:
                            poly_trial = self.optimizer.optimize(poly_trial)
                        except Exception as e:
                            logger.warning(
                                f'Optimizer failed at step {step+1}, attempt {attempt+1}: {e}'
                            )
                            continue
                        if poly_trial is None:
                            logger.warning(
                                f'Optimizer returned no molecule at step {step+1}, attempt {attempt+1}'
                            )
                            continue

                    # Validate the 3D structure
                    check = poly.check_3d_structure_poly(
                        poly_trial, piece,
                        dist_min=self.dist_min,
                        check_bond_length=self.check_bond_length,
                        added_atom_indices=added_indices,
                        removed_head_idx=removed_head_idx
                    )

                    if check:
                        poly_mol = poly_trial
                        poly_history.append(utils.deepcopy_mol(poly_mol))
                        if len(poly_history) > history_limit:
                            poly_history.pop(0)
                            history_start_step += 1
                        success = True

                        # Log progress at ~10% intervals (or every step for short chains)
                        progress = (step + 1) / (self.n - 1)
                        if (self.n - 1) <= 20 or int(progress * 10) > int(step / (self.n - 1) * 10):
                            elapsed = time.time() - t0
                            logger.info(
                                f'Step {step+1:3d}/{self.n-1} '
                                f'({progress*100:5.1f}%) [{elapsed:6.1f}s elapsed]'
                            )
                        break

                if success:
                    step += 1
                else:
                    logger.warning(f'Failed at step {step+1} after {self.retry_step} attempts')
                    success_chain = False
                    break

            if success_chain:
                break
            else:
                if retries_left <= 0:
                    logger.error('Failed to build polymer: all retries exhausted')
                    self._last_failure_reason = (
                        f'Could not build chain at step {step + 1} after '
                        f'{self.retry_step} attempts per step and {self.retry} retries'
                    )
                    return None

                retries_left -= 1
                total_rollbacks += 1
                self._last_rollbacks += 1
                rollback_steps = min(self.rollback, step - history_start_step)
                start_step = step - rollback_steps
                history_index = start_step - history_start_step

                logger.info(
                    f'Rollback {rollback_steps} steps -> step {start_step+1}. '
                    f'Retries left: {retries_left}'
                )

                poly_mol = utils.deepcopy_mol(poly_history[history_index])
                poly_history = poly_history[:history_index + 1]

        elapsed = time.time() - t0
        if poly_mol is not None:
            logger.info(
                f'Done in {elapsed:.1f}s. '
                f'Attempts: {total_attempts}, rollbacks: {total_rollbacks}'
            )
            logger.info(
                f'Atoms: {poly_mol.GetNumAtoms()}, bonds: {poly_mol.GetNumBonds()}'
            )
        else:
            logger.error(
                f'Failed after {elapsed:.1f}s. '
                f'Attempts: {total_attempts}, rollbacks: {total_rollbacks}'
            )

        return poly_mol

    def build(self):
        """Build a chain and return a :class:`BuildResult` with status and stats."""
        started = time.time()
        molecule = self.build_mol()
        return BuildResult(
            molecule=molecule,
            attempts=self._last_attempts,
            rollbacks=self._last_rollbacks,
            elapsed_seconds=time.time() - started,
            failure_reason=self._last_failure_reason,
        )


@dataclass(frozen=True)
class BuildResult:
    """Outcome and basic diagnostics from a polymerization run."""

    molecule: object
    attempts: int
    rollbacks: int
    elapsed_seconds: float
    failure_reason: str = None

    @property
    def success(self):
        return self.molecule is not None


def build_polymer(smiles, units, *, optimizer='mmff', optimizer_options=None,
                  config=None, seed=None):
    """Build a polymer directly from a monomer SMILES and repeat count."""
    polymerizer = Polymerizer.from_smiles(
        smiles,
        units,
        optimizer=optimizer,
        optimizer_options=optimizer_options,
        config=config,
        seed=seed,
    )
    return polymerizer.build()
