"""
Mathematical utilities for 3D geometry operations.

This module provides functions for vector math, rotations, dihedral angle
calculations, and distance matrix computations used throughout the
polymerization pipeline.
"""

import numpy as np
from numba import njit
from scipy.spatial.distance import cdist


@njit(cache=True)
def angle_vec(v1, v2, rad=False):
    """
    Compute the angle between two 3D vectors.

    Args:
        v1: First vector (3-element array-like).
        v2: Second vector (3-element array-like).
        rad: If True, return angle in radians; otherwise in degrees.

    Returns:
        Angle between v1 and v2 (float).
    """
    c = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    # Numba does not support np.clip for scalars, use manual bounds
    if c > 1.0:
        c = 1.0
    elif c < -1.0:
        c = -1.0
    a = np.arccos(c)
    return a if rad else np.degrees(a)


@njit(cache=True)
def rotate_rod(coords, axis, angle, center=None):
    """
    Rotate a set of 3D coordinates around an arbitrary axis (Rodrigues' formula).

    Args:
        coords: Nx3 array of atomic coordinates.
        axis: 3-element rotation axis vector (will be normalized).
        angle: Rotation angle in radians.
        center: Optional 3-element center of rotation. Defaults to origin.

    Returns:
        Nx3 array of rotated coordinates.
    """
    axis = axis / np.linalg.norm(axis)
    if center is None:
        center = np.zeros(3)
    coords = coords - center
    K = np.array([[0.0, -axis[2], axis[1]],
                  [axis[2], 0.0, -axis[0]],
                  [-axis[1], axis[0], 0.0]])
    R = np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)
    return (R @ coords.T).T + center


@njit(cache=True)
def dihedral_coord(p0, p1, p2, p3, rad=False):
    """
    Compute the dihedral angle defined by four 3D points (p0-p1-p2-p3).

    Uses the standard definition: angle between the planes (p0,p1,p2) and
    (p1,p2,p3), measured around the p1-p2 bond.

    Args:
        p0, p1, p2, p3: 3-element coordinate arrays.
        rad: If True, return angle in radians; otherwise in degrees.

    Returns:
        Dihedral angle (float) in the range (-pi, pi] or (-180, 180].
    """
    b1 = p1 - p0
    b2 = p2 - p1
    b3 = p3 - p2
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    b2n = b2 / np.linalg.norm(b2)
    m1 = np.cross(n1, b2n)
    x = np.dot(n1, n2)
    y = np.dot(m1, n2)
    d = np.arctan2(y, x)
    return d if rad else np.degrees(d)


def distance_matrix(coord1, coord2=None):
    """
    Compute pairwise Euclidean distance matrix between two sets of coordinates.

    Uses scipy.spatial.distance.cdist for optimized C-level performance.

    Args:
        coord1: Nx3 array of coordinates.
        coord2: Optional Mx3 array. If None, computes self-distance matrix.

    Returns:
        NxM (or NxN) distance matrix.
    """
    if coord2 is None:
        return cdist(coord1, coord1)
    return cdist(coord1, coord2)