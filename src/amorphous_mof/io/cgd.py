from __future__ import annotations

import os
from typing import Any

import numpy as np
import numpy.typing as npt
from pymatgen.core.lattice import Lattice as PymatgenLattice


def parse_cgd(filename: str):
    """
    Parses a CGD file and extracts lattice, atoms, and edges.

    Parameters
    ----------
    filename
        The path to the CGD file.
    
    Returns
    -------
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File '{filename}' not found.")

    lattice = None
    all_coords = []
    atom_labels = []
    edges = []

    with open(filename) as file:

        for line in file:
            tokens = line.split()
            if not tokens:
                continue

            match tokens[0]:

                case "cell":
                    lattice = np.array(
                        list(map(float, tokens[1:])), 
                        dtype=np.float64
                    )

                case "atom":
                    label, coords = tokens[1], list(map(float, tokens[3:]))
                    atom_labels.append(label)
                    all_coords.append(np.array(coords, dtype=np.float64))

                case "edge":
                    edges.append([tokens[1]] + list(map(float, tokens[2:5])))

    all_coords = np.array(all_coords) if all_coords else None
    lattice = PymatgenLattice.from_parameters(*lattice)

    return lattice, atom_labels, all_coords, edges


def process_neighbour_list(
    edges: list[list[Any]],
    atom_frac_coords: npt.NDArray[np.floating] | None,
    atom_labels: list[str],
) -> npt.NDArray[np.int_]:
    """
    Processes edges and determines node connectivity.

    Parameters
    ----------
    edges
        The list of edges in the network.
    atom_frac_coords
        The fractional coordinates of the atoms.
    atom_labels
        The labels of the atoms.

    Returns
    -------
    The neighbour list of the network, where each row corresponds to a pair of
    connected nodes and the applied image shift to the second node.
    """
    
    # create a mapping from atom label to index for fast lookups
    label_to_index = {label: i + 1 for i, label in enumerate(atom_labels)}

    neighbour_list = []
    for edge in edges:

        node_label, frac_coords = edge[0], np.array(edge[1:], dtype=np.float64)

        if node_label not in label_to_index:
            raise ValueError(f"Unrecognised atom label in edge: {node_label}")

        wrapped_coords = frac_coords % 1.0
        img = (frac_coords - wrapped_coords).astype(int)

        dists = (
            np.linalg.norm(atom_frac_coords - wrapped_coords, axis=1)
            if atom_frac_coords is not None else None
        )
        closest_atom_idx = np.argmin(dists) if dists is not None else None

        if dists is not None and dists[closest_atom_idx] > 1e-4:
            raise ValueError(f"Could not find closest atom to edge {edge}.")

        node1 = label_to_index[node_label]
        node2 = closest_atom_idx + 1 if closest_atom_idx is not None else None

        if node2 is not None:
            neighbour_list.append([node1, node2, *img])

    neighbour_list = (
        np.array(neighbour_list, dtype=int) if neighbour_list 
        else np.empty((0, 5), dtype=int)
    )

    return neighbour_list