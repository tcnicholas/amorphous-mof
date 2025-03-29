from __future__ import annotations

from typing import NamedTuple, Sequence

import numpy as np
import numpy.typing as npt

from amorphous_mof.utils import uniform_repr


class VertexSymbol(NamedTuple):

    vector: list[list[int]]
    vector_all_rings: list[list[int]]

    def __repr__(self) -> str:
        info = {} 
        info["VS"] = self.to_str()
        info["VS(all_rings)"] = self.to_str(all_rings=True)
        return uniform_repr("VertexSymbol", **info, indent_size=4)

    def to_str(self, all_rings: bool = False) -> str:
        """
        Returns the string representation of the VertexSymbol.

        If `all_rings` is True, ring counts are grouped and formatted with 
        multiplicity. Otherwise, only the smallest ring sizes are shown, with 
        multiplicity for repeated values.

        Parameters
        ----------
        all_rings
            If True, ring counts are grouped and formatted with multiplicity.
            Otherwise, only the smallest ring sizes are shown, with multiplicity
            for repeated values.
        
        Returns
        -------
        A string representation of the VertexSymbol.
        """

        vector = self.vector_all_rings if all_rings else self.vector

        formatted_elements = []
        for rings in vector:

            ring_counts = {size: rings.count(size) for size in set(rings)}

            # Format elements with multiplicity if needed
            element = ",".join(
                f"{size}({count})" if count > 1 else f"{size}"
                for size, count in sorted(ring_counts.items())
            )

            # Use parentheses only if there are multiple distinct ring sizes
            if len(set(rings)) > 1:
                formatted_elements.append(f"({element})")
            else:
                formatted_elements.append(element)

        return f"[{'.'.join(formatted_elements)}]"


class CARVS(NamedTuple):
    """
    Cummulative All-Rings Vertex Symbol (CARVS) vector.

    Attributes
    ----------
    vector
        The CARVS vector.
    spread
        The standard deviation of the CARVS vectors in the network.
    is_single_node
        True if the CARVS vector is for a single-node network, False otherwise.
    """

    vector: npt.NDArray[np.floating]
    spread: float
    is_single_node: bool


    @classmethod
    def from_list(cls, carvs_list: Sequence[CARVS]) -> CARVS:
        """
        Construct a CARVS object from a list of CARVS objects,
        averaging the vectors and spreads.

        Parameters
        ----------
        carvs_list
            One or more CARVS objects to be averaged.

        Returns
        -------
        A new CARVS object whose vector is the average of all input vectors, 
        whose spread is the average of all input spreads, and whose 
        'is_single_node' is True only if it is True for every entry in 
        `carvs_list`.

        Raises
        ------
        ValueError
            If `carvs_list` is empty or if the vectors in `carvs_list` do not 
            all have the same length.
        """

        if not carvs_list:
            raise ValueError("Cannot create a CARVS from an empty list.")
        
        # 1. pad the vectors to the same length.
        padded_carvs = pad_carvs(carvs_list)

        # 2. average the vectors.
        padded_vectors_array = np.array([c.vector for c in padded_carvs])
        avg_vector = padded_vectors_array.mean(axis=0)

        # 3. average the spreads
        avg_spread = float(np.mean([c.spread for c in carvs_list]))

        # 4. decide how to set is_single_node
        all_single_node = all(c.is_single_node for c in carvs_list)

        return cls(
            vector=avg_vector,
            spread=avg_spread,
            is_single_node=all_single_node,
        )

    def __str__(self) -> str:
        """Generate a formatted string representation of the object."""
        
        lbracket, rbracket = "{", "}"
        elements = []

        for size, count in enumerate(self.vector, 1):
            if count < 1.0:
                continue
            if count == 1.0:
                elements.append(f"{size}.")
            else:
                formatted_count = (
                    f"{int(round(count))}" if abs(count - round(count)) < 1e-5 
                    else f"{count:.1f}"
                )
                elements.append(f"{size}({formatted_count}).")

        symbol = lbracket + "".join(elements).rstrip('.') + rbracket

        if not np.isclose(self.spread, 0.0):
            symbol += f" σ={self.spread:.1f}"

        return symbol
    
    def __repr__(self) -> str:
        """Generate a string representation of the object."""
    
        return f"CARVS( {str(self)} )"
    

############################### HELPERS ###############################


def pad_carvs(carvs_list: Sequence[CARVS]) -> Sequence[CARVS]:
    """
    Pad the vectors of a list of CARVS objects to the same length.
    
    Parameters
    ----------
    carvs_list
        A list of CARVS objects.

    Returns
    -------
    A list of CARVS objects with the vectors padded to the same length.
    """

    max_length = max(len(carv.vector) for carv in carvs_list)

    padded_vectors = []
    for carvs in carvs_list:
        padded = np.zeros(max_length, dtype=float)
        padded[:len(carvs.vector)] = carvs.vector
        padded_vectors.append(padded)

    new_carvs = []
    for orig_carvs, padded_vector in zip(carvs_list, padded_vectors):
        new_carvs.append(
            CARVS(
                vector=padded_vector, 
                spread=orig_carvs.spread,
                is_single_node=orig_carvs.is_single_node,
            )
        )
    
    return new_carvs


def pad_carvs_per_atom(
    all_carvs: list[npt.NDArray[np.int_]]
) -> list[npt.NDArray[np.int_]]:
    """ 
    Pad the CARVs per atom to the same length.
    
    Parameters
    ----------
    all_carvs
        List of CARVs per atom.
        
    Returns
    -------
    List of padded CARVs per atom.
    """

    max_length = max(c.shape[1] for c in all_carvs)

    padded_carvs = [
        np.pad(c, ((0, 0), (0, max_length - c.shape[1])), mode="constant")
        for c in all_carvs
    ]

    return padded_carvs


############################### ANALYSIS ###############################


def get_all_topological_distances(carvs: list[CARVS]) -> np.ndarray:
    """
    Compute the topological distances between all pairs of CARVS objects.

    Parameters
    ----------
    carvs
        A list of CARVS objects.

    Returns
    -------
    A square matrix of shape (n_points, n_points) containing the Euclidean 
    distances between all pairs of points.
    """

    if not isinstance(carvs, list):
        raise TypeError("'carvs' must be a list of CARVS objects")
    
    if not all(isinstance(carv, CARVS) for carv in carvs):
        raise TypeError("All elements of 'carvs' must be CARVS objects")

    # 1. gather the vectors of all carvs objects.
    carvs_vectors = np.array([carvs.vector for carvs in pad_carvs(carvs)])

    # 2. divide by sum of values.
    carvs_vectors /= carvs_vectors.sum(axis=1)[:, np.newaxis]

    # 3. compute distance between all pairs of vectors as:
    #
    #   d(\alpha, \beta) = \frac{1}{\sqrt{2}} | r(\alpha) - r(\beta) |
    #
    distances = np.sqrt(
        np.maximum(
            np.einsum('ij,ij->i', carvs_vectors, carvs_vectors)[:, None]
            + np.einsum('ij,ij->i', carvs_vectors, carvs_vectors)[None, :]
            - 2 * np.einsum('ik,jk->ij', carvs_vectors, carvs_vectors),
            0
        )
    )

    return distances / np.sqrt(2)