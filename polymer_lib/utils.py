"""
Utility functions for RDKit Mol manipulation and logging.

Provides helpers for atom/bond editing, SMILES conversion, deep copying
of Mol objects with full property preservation, and a simple logger.
"""

import re
from copy import deepcopy
from rdkit import Chem
from rdkit.Chem import AllChem
import logging
import sys


# ============================================================
# Logging
# ============================================================


def get_logger(name="polymer_lib"):
    """
    Create or retrieve a configured logger.

    Args:
        name: Logger name (default: 'polymer_lib').

    Returns:
        A configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


logger = get_logger()


# ============================================================
# Mol manipulation
# ============================================================


def deepcopy_mol(mol):
    """
    Deep-copy an RDKit Mol object preserving all properties.

    Args:
        mol: RDKit Mol object.

    Returns:
        A deep copy of the input Mol.
    """
    Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)
    return deepcopy(mol)


def remove_atom(mol, idx):
    """
    Remove an atom and all its incident bonds from a Mol.

    Args:
        mol: RDKit Mol object.
        idx: Index of the atom to remove.

    Returns:
        A new Mol object with the atom removed.
    """
    rwmol = Chem.RWMol(mol)
    for pb in mol.GetAtomWithIdx(idx).GetNeighbors():
        rwmol.RemoveBond(idx, pb.GetIdx())
    rwmol.RemoveAtom(idx)
    return rwmol.GetMol()


def add_bond(mol, idx1, idx2, order=Chem.rdchem.BondType.SINGLE):
    """
    Add a bond between two atoms in a Mol.

    Args:
        mol: RDKit Mol object.
        idx1, idx2: Indices of the atoms to connect.
        order: Bond type (default: SINGLE).

    Returns:
        A new Mol object with the added bond.
    """
    rwmol = Chem.RWMol(mol)
    rwmol.AddBond(idx1, idx2, order=order)
    return rwmol.GetMol()


# ============================================================
# SMILES <-> Mol conversion
# ============================================================


def star2h(smiles):
    """
    Replace dummy atoms ('*') in SMILES with tritium ('[3H]').

    This allows RDKit to embed 3D coordinates for the linker atoms,
    which it otherwise refuses to do for dummy atoms.

    Args:
        smiles: Input SMILES string containing '*' atoms.

    Returns:
        SMILES string with '*' replaced by '[3H]' (or '[nH]' for numbered dummies).
    """
    smiles = smiles.replace("[*]", "[3H]")
    smiles = re.sub(
        r"\[([0-9]+)\*\]", lambda m: "[%iH]" % int(int(m.groups()[0]) + 2), smiles
    )
    smiles = smiles.replace("*", "[3H]")
    return smiles


def h2star(smiles):
    """
    Reverse of star2h: replace tritium ('[3H]') back to dummy atoms ('*').

    Args:
        smiles: SMILES string containing '[3H]' atoms.

    Returns:
        SMILES string with '[3H]' replaced back by '*'.
    """
    smiles = smiles.replace("[3H]", "*")
    smiles = re.sub(
        r"\[([0-9]+)H\]",
        lambda m: "[%i*]" % int(int(m.groups()[0]) - 2)
        if int(int(m.groups()[0])) >= 3
        else "[%iH]" % int(int(m.groups()[0])),
        smiles,
    )
    return smiles


def mol_from_smiles(smiles, coord=True):
    """
    Build an RDKit Mol with 3D coordinates from a SMILES string.

    Dummy atoms ('*') are temporarily converted to tritium for embedding,
    then converted back.

    Args:
        smiles: Input SMILES (may contain '*' for connection points).
        coord: If True, generate 3D coordinates via ETKDGv3.

    Returns:
        RDKit Mol object, or None on failure.
    """
    smi = star2h(smiles)
    etkdg = AllChem.ETKDGv3()
    etkdg.enforceChirality = True
    etkdg.useRandomCoords = False
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            logger.error(f'Cannot parse SMILES: {smiles}')
            return None
        mol = Chem.AddHs(mol)
    except Exception as e:
        logger.error(f'Cannot transform to RDKit Mol object from {smiles}: {e}')
        return None
        
    if coord:
        try:
            enbed_res = AllChem.EmbedMolecule(mol, etkdg)
        except Exception as e:
            logger.error(f'Cannot generate 3D coordinate of {smiles}: {e}')
            return None
            
        if enbed_res == -1:
            etkdg.useRandomCoords = True
            try:
                enbed_res = AllChem.EmbedMolecule(mol, etkdg)
            except Exception as e:
                logger.error(f'Cannot generate 3D coordinate of {smiles}: {e}')
                return None
            if enbed_res == -1:
                logger.error(f'Cannot generate 3D coordinate of {smiles}')
                return None
    return mol
