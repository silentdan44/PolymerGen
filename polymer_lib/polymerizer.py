"""
Polymerization controller.

The Polymerizer class orchestrates the step-by-step assembly of a polymer
chain from a list of Monomer objects, using retry and rollback logic to
recover from steric clashes.
"""

import time
import numpy as np
from . import utils
from . import poly
from .utils import logger
from .optimizers import BaseOptimizer, get_optimizer


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
                 dist_min=0.7, retry=100, rollback=5, retry_step=200,
                 check_bond_length=True, optimizer=None):
        """
        Initialize the Polymerizer.

        Args:
            monomers: List of Monomer objects.
            bond_length: Target length for new bonds (Angstrom).
            dihedral: Target dihedral angle around new bonds (radians).
            random_rot: If True, use a random dihedral at each step.
            dist_min: Minimum allowed non-bonded distance (Angstrom).
            retry: Max number of full-chain rollbacks before giving up.
            rollback: Number of steps to roll back on failure.
            retry_step: Max attempts per polymerization step.
            check_bond_length: If True, also validate bond lengths.
            optimizer: Structure optimizer. Can be:
                - None (no optimization)
                - A BaseOptimizer instance (e.g. MMFFOptimizer)
                - A string ('mmff' or 'openff')
                - A dict like {'name': 'mmff', 'max_iters': 100}
        """
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

        # Parse optimizer into a BaseOptimizer instance
        self.optimizer = self._parse_optimizer(optimizer)

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

    def build(self):
        """
        Run the polymerization process and return the final polymer Mol.

        Returns:
            RDKit Mol object of the polymer with 3D coordinates, or
            None if the build failed after exhausting all retries.
        """
        if self.n < 1:
            raise ValueError('Number of monomers must be >= 1')

        t0 = time.time()
        logger.info(f'Start polymerization: n={self.n}')
        if self.optimizer is not None:
            logger.info(f'Optimizer: {self.optimizer.name}')

        poly_mol = utils.deepcopy_mol(self.monomers[0].mol)
        poly_history = [utils.deepcopy_mol(poly_mol)]

        total_attempts = 0
        total_rollbacks = 0
        retries_left = self.retry
        start_step = 0

        while start_step < self.n - 1:
            step = start_step
            success_chain = True

            while step < self.n - 1:
                piece = utils.deepcopy_mol(self.monomers[step + 1].mol)
                success = False

                for attempt in range(self.retry_step):
                    total_attempts += 1
                    poly_trial = poly.connect_mols(
                        poly_mol, piece,
                        bond_length=self.bond_length,
                        dihedral=self.dihedral,
                        random_rot=self.random_rot
                    )

                    # Optional structure optimization after connection
                    if self.optimizer is not None:
                        poly_trial = self.optimizer.optimize(poly_trial)

                    # Validate the 3D structure
                    check = poly.check_3d_structure_poly(
                        poly_trial, piece,
                        dist_min=self.dist_min,
                        check_bond_length=self.check_bond_length
                    )

                    if check:
                        poly_mol = poly_trial
                        poly_history.append(utils.deepcopy_mol(poly_mol))
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
                    return None

                retries_left -= 1
                total_rollbacks += 1
                rollback_steps = min(self.rollback, len(poly_history) - 1)
                start_step = max(0, step - rollback_steps)

                logger.info(
                    f'Rollback {rollback_steps} steps -> step {start_step+1}. '
                    f'Retries left: {retries_left}'
                )

                poly_mol = utils.deepcopy_mol(poly_history[start_step])
                poly_history = poly_history[:start_step + 1]

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