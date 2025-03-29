import numpy as np
from numba import njit
from typing import Union


@njit
def deg2rad(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Convert degrees to radians.

    Parameters:
    ----------
    x : float or np.ndarray
        Angle(s) in degrees.

    Returns:
    -------
    float or np.ndarray
        Angle(s) converted to radians.
    """
    return x * np.pi / 180


@njit
def rad2deg(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Convert radians to degrees.

    Parameters:
    ----------
    x : float or np.ndarray
        Angle(s) in radians.

    Returns:
    -------
    float or np.ndarray
        Angle(s) converted to degrees.
    """
    return x * 180 / np.pi


@njit
def unit_vector(v: np.ndarray) -> np.ndarray:
    """
    Normalize a 1x3 vector to a unit vector.

    Parameters:
    ----------
    v : np.ndarray
        Input vector of shape (3,).

    Returns:
    -------
    np.ndarray
        Normalized unit vector of shape (3,).
    """
    return v / np.sqrt(np.sum(np.square(v)))


@njit
def random_vector(rand1: float, rand2: float) -> np.ndarray:
    """
    Generate a random unit vector using two random numbers.

    Parameters:
    ----------
    rand1 : float
        A random float in the range [0, 1].
    rand2 : float
        A random float in the range [0, 1].

    Returns:
    -------
    np.ndarray
        A random unit vector of shape (3,).
    """
    phi = rand1 * 2 * np.pi  # Random azimuthal angle in radians.
    z = rand2 * 2 - 1  # Random z-coordinate in the range [-1, 1].

    z2 = z * z
    x = np.sqrt(1 - z2) * np.cos(phi)
    y = np.sqrt(1 - z2) * np.sin(phi)

    return np.array([x, y, z])


@njit
def rodrigues(a: np.ndarray, b: np.ndarray, theta: float) -> np.ndarray:
    """
    Apply Rodrigues' rotation formula to rotate a vector about another vector.

    Parameters:
    ----------
    a : np.ndarray
        The vector to rotate of shape (3,).
    b : np.ndarray
        The rotation axis vector of shape (3,). Must be a unit vector.
    theta : float
        The rotation angle in radians.

    Returns:
    -------
    np.ndarray
        The rotated vector of shape (3,).
    """
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    term1 = a * cos_theta
    term2 = np.cross(b, a) * sin_theta
    term3 = b * np.dot(b, a) * (1 - cos_theta)

    return term1 + term2 + term3
