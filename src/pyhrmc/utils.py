import itertools
from ctypes import c_int, c_double
from typing import List, Tuple, Union

import numpy as np
from numba import njit, prange


def unique_pairs(l: List[int]) -> List[Tuple[int, int]]:
    """
    Generate unique pairs from a 1D list, including self-pairs.

    Parameters:
    ----------
    l : List[int]
        Input list of integers.

    Returns:
    -------
    List[Tuple[int, int]]
        List of unique pairs (including self-pairs) sorted by the first element.
    """
    pairs = list(itertools.combinations(l, r=2)) + [(x, x) for x in l]
    return sorted((sorted(x) for x in pairs), key=lambda x: x[0])


def round_maxr(max_r: float, bin_width: float) -> float:
    """
    Round the maximum radius value by flooring it to multiple of the bin width.

    Parameters:
    ----------
    max_r : float
        The maximum radius value to round.
    bin_width : float
        The bin width to floor `max_r` to.

    Returns:
    -------
    float
        The rounded maximum radius.
    """
    try:
        rv = len(str(bin_width).split(".")[-1])
    except:
        rv = 0

    top_bin = bin_width * np.floor(max_r / bin_width)
    return np.round(top_bin, rv)


def ids2cids(ids: List[int]) -> c_int:
    """
    Convert a list of integers to a ctypes array of c_int.

    Parameters:
    ----------
    ids : List[int]
        List of integers.

    Returns:
    -------
    ctypes array of c_int
        ctypes array representation of the input integers.
    """
    lenids = len(ids)
    cids = (lenids * c_int)()
    for i in range(lenids):
        cids[i] = ids[i]
    return cids


def cdata2pydata(cdata: Union[c_int, c_double], dim: int) -> np.ndarray:
    """
    Convert ctypes data to a NumPy array.

    Parameters:
    ----------
    cdata : ctypes array
        Input ctypes array.
    dim : int
        Dimensionality of the resulting array.

    Returns:
    -------
    np.ndarray
        Converted NumPy array.
    """
    cdata = list(cdata)
    if dim > 1:
        return np.array([cdata[x : x + dim] for x in range(0, len(cdata), dim)])
    else:
        return np.array(cdata)


def numpy2c(pyVector: np.ndarray, gtype: int) -> Union[c_int, c_double]:
    """
    Convert a NumPy array to a ctypes array.

    Parameters:
    ----------
    pyVector : np.ndarray
        Input NumPy array.
    gtype : int
        Type of conversion:
        0 -> c_int
        1 -> c_double

    Returns:
    -------
    ctypes array
        Converted ctypes array.
    """
    pyVector = np.array(pyVector).flatten()
    if gtype == 0:
        return (len(pyVector) * c_int)(*pyVector)
    else:
        return (len(pyVector) * c_double)(*pyVector)


@njit(parallel=True)
def mask_distances(d: np.ndarray, nmax: int, tol: float, rcut: float) -> float:
    """
    Compute the masked sum of distances based on a given tolerance and cutoff.

    Parameters:
    ----------
    d : np.ndarray
        Array of distances.
    nmax : int
        Maximum number of distances to consider.
    tol : float
        Tolerance threshold for distances.
    rcut : float
        Cutoff radius.

    Returns:
    -------
    float
        Sum of contributions where distances satisfy the tolerance and cutoff.
    """
    total_v = 0.0
    for n in prange(nmax):
        if d[n] >= tol:
            v = rcut - d[n]
            if v > 0:
                total_v += v
    return total_v


def append_array_to_file(arr, filename):
    with open(filename, "ab") as f:
        np.savetxt(f, arr.reshape(1, arr.size), delimiter=",", newline="\n")
