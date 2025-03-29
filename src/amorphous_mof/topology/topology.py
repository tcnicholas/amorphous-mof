from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, NamedTuple

import numpy as np
import numpy.typing as npt
from pymatgen.core.lattice import Lattice as PymatgenLattice

from amorphous_mof.io.cgd import parse_cgd, process_neighbour_list
from amorphous_mof.paths import RingStatistics as RS
from amorphous_mof.topology.clusters import (
    Cluster,
    RingSizeCounts,
    get_carvs_vector,
    get_clusters,
    get_vertex_symbol,
)
from amorphous_mof.utils import uniform_repr


class RingsResults(NamedTuple):

    depth: int
    """ The depth to which the rings were searcher for. """

    rings_are_strong: bool
    """ Whether the rings were filtered to strong rings only. """

    ring_size_count: RingSizeCounts
    """
    The number of rings of a given size, of shape ``(**, 2)`` where the first
    column indicates the ring size, and the second indicates the number of rings
    of that size.
    """

    clusters: list[Cluster]
    """
    Each node can be characterised in terms of the rings in which it 
    participates. This can be summarised using common metrics such as Vertex 
    Symbols.
    """
    
    def __repr__(self) -> str:


        vs_name = "VertexSymbol"
        vertex_symbols_str = ""

        vertex_symbols = {x.to_str() for x in get_vertex_symbol(self.clusters)}
    
        if len(vertex_symbols) > 1:
            vs_name = "VertexSymbols"
            vertex_symbols_str += "{\n\t"

        vertex_symbols_str += ",\n\t".join([f"{x}" for x in vertex_symbols])

        if len(vertex_symbols) > 1:
            vertex_symbols_str += "\n}"
    
        info = {
            "depth": self.depth,
            "strong_rings": self.rings_are_strong,
            "ring_size_count": self.ring_size_count,
            vs_name: vertex_symbols_str,
            "CARVS": get_carvs_vector(self.clusters),
        }

        return uniform_repr(
            "RingsResults", **info, indent_size=4, stringify=False
        )

@dataclass
class Topology:
    """
    A class detailing the topology of a network, based on nodes and edges.
    """

    nodes: list[Node]
    edges: npt.NDArray[np.int_] = field(
        default_factory=lambda: np.empty((0, 5), dtype=int)
    )
    lattice: PymatgenLattice | None = None
    properties: dict[str, dict[str, Any]] = field(default_factory=dict)


    @classmethod
    def from_cgd(cls, filename: str) -> Topology:
        """
        Parses and loads a CGD file with an adjacency matrix.

        Parameters
        ----------
        filename
            The path to the CGD file.

        Returns
        -------
        A Topology object representing the network as nodes and edges.
        """

        if not os.path.exists(filename):
            raise FileNotFoundError(f"File '{filename}' not found.")
        
        lattice, atom_labels, atoms, edges = parse_cgd(filename)
        neighbour_list = process_neighbour_list(edges, atoms, atom_labels)

        # Create Node instances
        all_nodes = [
            Node(
                node_id=i + 1, 
                node_type=label,
                frac_coord=frac_coord if atoms is not None else None
            )
            for i, (label, frac_coord) in enumerate(zip(atom_labels, atoms))
            if atoms is not None
        ]

        return cls(nodes=all_nodes, edges=neighbour_list, lattice=lattice)

    def get_rings(self, depth: int = 12, strong: bool = False) -> RingsResults:
        """Computes or retrieves ring statistics for the network.
        
        Parameters
        ----------
        depth
            The maximum depth to search for rings.
        strong
            Whether to filter the rings to strong rings only.

        Returns
        -------
        A dictionary containing the ring statistics.
        """

        label = ("strong_rings" if strong else "rings", depth)

        # check if cached results exist and return them.
        if label in self.properties:
            return self.properties[label]

        # compute rings.
        compute_rings = RS.run_strong_rings if strong else RS.run_rings
        rcount, _, rnodes = compute_rings(self.edges, depth)

        # store and return results.
        results = RingsResults(
            depth=depth,
            rings_are_strong=strong,
            ring_size_count=RingSizeCounts(*rcount.T),
            clusters=get_clusters(self, rnodes),
        )

        self.properties[label] = results

        return results
    
    def get_topological_genome(self) -> str:
        """ Returns a string representation of the network topology. """

        if "topological_genome" in self.properties:
            return self.properties["topological_genome"]
    
        nodes = get_all_node_frac_coords(self.nodes)
        cell_lengths = self.lattice.lengths
        cell_angles = self.lattice.angles

        topology_genome = RS.get_topological_genome(
            nodes.T, # transpose to get shape (3, N) for Julia. 
            self.edges, 
            cell_lengths, 
            cell_angles
        )

        self.properties["topological_genome"] = topology_genome

        return topology_genome

    def __getitem__(self, node_id: int) -> Node:
        """Retrieve a Node object by its node number."""
        if not (1 <= node_id <= len(self.nodes)):
            raise IndexError(
                f"node_id {node_id} is out of bounds "
                f"(valid range: 1 to {len(self.nodes)})"
            )

        return self.nodes[node_id - 1]

    def __repr__(self) -> str:
        info = {
            "nodes": len(self.nodes),
            "edges": len(self.edges),
            "has_lattice": self.lattice is not None,
        }
        return uniform_repr("Topology", **info, indent_size=4)


@dataclass(order=True)
class Node:
    """
    A representation of a node in a network.
    """

    node_id: int
    node_type: str | None = "Si"
    frac_coord: npt.NDArray[np.floating] | None = field(default=None)
    cart_coord: npt.NDArray[np.floating] | None = field(default=None)

    is_shifted: bool = field(default=False)

    def apply_image_shift(
        self,
        lattice: PymatgenLattice,
        image_shift: npt.NDArray[np.int_],
    ) -> Node:
        """Apply the image shift to this node and return a new Node object. 
        
        Parameters
        ----------
        lattice
            The lattice object for the network.
        image_shift
            The shift vector to apply to the node coordinates.

        Returns
        -------
        A new Node object with the shifted coordinates.
        """
        
        if self.frac_coord is None and self.cart_coord is None:
            raise ValueError(
                "Both `frac_coord` and `cart_coord` are missing; "
                "cannot compute shifted coordinates."
            )
        
        if self.frac_coord is not None:
            frac_coord = self.frac_coord
        else:
            frac_coord = lattice.get_fractional_coords(self.cart_coord)
        
        shifted_frac = frac_coord + image_shift
        shifted_cart = lattice.get_cartesian_coords(shifted_frac)
        
        return Node(
            node_id=self.node_id,
            node_type=self.node_type,
            frac_coord=shifted_frac,
            cart_coord=shifted_cart,
            is_shifted=True
        )
    
    def __post_init__(self) -> None:
        """ Ensure coordinates are NumPy arrays if provided. """
        self.sort_index = self.node_id
        if self.frac_coord is not None:
            self.frac_coord = np.array(self.frac_coord, dtype=np.float64)
        if self.cart_coord is not None:
            self.cart_coord = np.array(self.cart_coord, dtype=np.float64)

    def __repr__(self) -> str:

        name = "Node" if not self.is_shifted else "ShiftedNode"
        info = {"node_id": self.node_id, "node_type": self.node_type}

        if self.frac_coord is not None:
            formatted_coords = np.array2string(
                np.round(self.frac_coord, 2), 
                precision=2, 
                separator=", ", 
                floatmode="fixed"
            )
            info["frac_coord"] = formatted_coords

        if self.cart_coord is not None:
            formatted_coords = np.array2string(
                np.round(self.cart_coord, 2),
                precision=2,
                separator=", ",
                floatmode="fixed"
            )
            info["cart_coord"] = formatted_coords

        return uniform_repr(name, **info, indent_size=4)


############################### HELPERS ###############################

def get_all_node_frac_coords(nodes: list[Node]) -> npt.NDArray[np.floating]:
    """Return the fractional coordinates of all nodes in the network."""

    return np.array([node.frac_coord for node in nodes], dtype=np.float64)