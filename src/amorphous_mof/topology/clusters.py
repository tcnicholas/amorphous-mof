from __future__ import annotations

import warnings
from collections import defaultdict
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import numpy.typing as npt

from amorphous_mof.topology.symbols import CARVS, VertexSymbol
from amorphous_mof.utils import to_tuple, uniform_repr

if TYPE_CHECKING:
    from amorphous_mof.topology.topology import Node, Topology


class RingSizeCounts(NamedTuple):

    sizes: npt.NDArray[np.int_]
    counts: npt.NDArray[np.int_]

    def __repr__(self) -> str:

        nonzero_idx = np.where(self.counts > 0)[0]

        if len(nonzero_idx) == 0:
            # Edge case: no rings or all counts are zero.
            info = {
                "n_rings": 0,
                "min_ring_size": None,
                "max_ring_size": None,
            }
        else:
            # Minimum and maximum sizes where counts are non-zero
            min_ring_size = int(self.sizes[nonzero_idx].min())
            max_ring_size = int(self.sizes[nonzero_idx].max())

            info = {
                "n_rings": int(self.counts.sum()),
                "min_ring_size": min_ring_size,
                "max_ring_size": max_ring_size,
            }
    
        return uniform_repr("RingSizeCounts", **info, indent_size=4)


class Ring(NamedTuple):

    nodes: list[Node]
    """ The node IDs that form the ring. Neighbouring nodes are connected by an 
    edge, and the last node is connected to the first node. """

    angle: tuple[tuple[int, tuple[int]]]
    """ 
    Neighbouring nodes about the central node, listed with node ID and image. 
    """

    @property
    def size(self) -> int:
        """ Ring size. """
        return len(self)

    def __len__(self) -> int:
        """ Ring size. """
        return len(self.nodes)

    def __repr__(self) -> str:
        info = {"n": len(self.nodes)}
        return uniform_repr("Ring", **info, indent_size=4)
    

class Cluster(NamedTuple):

    central_node_id: int
    """ The node ID of the central node in the cluster. """

    rings: list[Ring]
    """ The rings that form the cluster. """

    def __repr__(self) -> str:
        info = {
            "node_id": self.central_node_id,
            "n_rings": len(self.rings),
            "CARVS": str(get_carvs_vector(self)),
        }
        return uniform_repr("Cluster", **info, indent_size=4)


############################### PROPERTIES ###############################


def smallest_ring_size(cluster: Cluster) -> int:
    """
    Get the smallest ring size in the ``cluster``.
    """

    return min(len(ring) for ring in cluster.rings)


def largest_ring_size(cluster: Cluster) -> int:
    """
    Get the largest ring size in the ``cluster``.
    """

    return max(len(ring) for ring in cluster.rings)


def number_of_rings(cluster: Cluster) -> int:
    """
    Get the number of rings in the ``cluster``.
    """

    return len(cluster.rings)


def get_unique_angles(cluster: Cluster) -> int:
    """
    Get the number of angles in the ``cluster``.
    """

    return list(set(ring.angle for ring in cluster.rings))


def number_of_angles(cluster: Cluster) -> int:
    """
    Get the number of angles in the ``cluster``.
    """

    return len(get_unique_angles(cluster))


def get_ring_sizes(cluster: Cluster) -> list:
    """
    Get the ring sizes in the ``cluster``.
    """

    return sorted([ring.size for ring in cluster.rings])


############################### HELPERS ###############################


def shares_edge(angle1: tuple, angle2: tuple) -> bool:
    """
    Checks if two angles share an edge.

    Parameters
    ----------
    angle1
        The first angle.
    angle2
        The second angle.

    Returns
    -------
    True if the angles share an edge, False otherwise.
    """

    return not set(angle1).isdisjoint(angle2)


def get_opposite_angles(angles: list) -> list:
    """
    Finds the opposite angles in a cluster.

    Parameters
    ----------
    angles
        A list of angles in the cluster.

    Returns
    -------
    A list of opposite angle pairs.
    """

    opposite_pairs = []
    remaining_angles = angles.copy()

    while remaining_angles:
        angle1 = remaining_angles.pop(0)
        for angle2 in remaining_angles:
            if not shares_edge(angle1, angle2):
                opposite_pairs.append((angle1, angle2))
                remaining_angles.remove(angle2)
                break
    
    return opposite_pairs


def only_smallest_ring_size(ring_sizes: list[int]) -> list[int]:
    """
    Retains only the smallest ring size in this list. If there are multiple
    smallest ring sizes, they are all retained.

    Parameters
    ----------
    ring_sizes
        A list of ring sizes.

    Returns
    -------
    A list of the smallest ring sizes.
    """
    
    return [size for size in ring_sizes if size == min(ring_sizes)]


def get_ring_size_by_angle(
    cluster: Cluster, 
    all_rings: bool = False,
) -> dict[int, list[int]]:
    """
    Get the ring sizes by angle in a cluster.

    Parameters
    ----------
    cluster
        The cluster to get the ring sizes by angle of.
    all_rings
        Whether to consider all rings in the cluster or just the smallest ring
        at each angle.

    Returns
    -------
    A dictionary of ring sizes by angle.
    """

    angle_to_rings = defaultdict(list)
    for ring in cluster.rings:
        angle_to_rings[ring.angle].append(ring.size)

    sorted_ring_sizes = {
        angle: sorted(sizes) for angle, sizes in angle_to_rings.items()
    }

    if not all_rings:
        sorted_ring_sizes = {
            angle: only_smallest_ring_size(sizes)
            for angle, sizes in sorted_ring_sizes.items()
        }

    sorted_ring_sizes = dict(
        sorted(
            sorted_ring_sizes.items(), 
            key=lambda item: tuple(item[1]),
        )
    )
    
    return sorted_ring_sizes


############################### CONVERSIONS ###############################


def get_ordered_node_list(
    nodes_and_images: list[tuple[int, npt.NDArray[np.int_]]],
    central_node_id: int
) -> list[tuple[int, npt.NDArray[np.int_]]]:
    """ 
    Sorts a cyclic list of nodes and images, placing the central_node_id first 
    and preserving cyclic order while ensuring a deterministic ordering.

    Parameters
    ----------
    nodes_and_images
        A list of node IDs and image shifts.
    central_node_id
        The node ID of the central node.
    
    Returns
    -------
    A sorted list of node IDs and image shifts.
    """

    def should_reverse(ring: list[tuple[int, npt.NDArray[np.int_]]]) -> bool:
        """ 
        Determines if the ring should be reversed by checking node ID order.
        """

        before = ring[-1]
        after = ring[1]

        if before[0] < after[0]:    # If previous ID is smaller, reverse
            return True
        elif before[0] > after[0]:  # If next ID is smaller, keep order
            return False

        # If node IDs are equal, compare their image values
        before = ring[-1] 
        after = ring[1]

        if tuple(before[1]) < tuple(after[1]):
            return False
        elif tuple(before[1]) > tuple(after[1]):
            return True

        return False

    central_node = None
    for node, image in nodes_and_images:
        if node == central_node_id and np.array_equal(image, [0, 0, 0]):
            central_node = (node, image)
            break

    if central_node is None:
        raise ValueError("Central node with image [0, 0, 0] not found")

    # find the index of the central node and rotate list.
    central_index = nodes_and_images.index(central_node)
    sorted_ring = nodes_and_images[central_index:] + \
        nodes_and_images[:central_index]

    # check if we need to reverse order.
    if should_reverse(sorted_ring):
        sorted_ring.reverse()
        sorted_ring = [sorted_ring[-1]] + sorted_ring[:-1]

    return sorted_ring


def node_list_to_ring(
    topology: Topology,
    node_list: list[tuple[int, npt.NDArray[np.int_]]],
    central_node_id: int
) -> Ring:
    """ Convert a list of node IDs and image shifts to a Ring object. 
    
    Parameters
    ----------
    topology
        The topology object containing the nodes.
    node_list
        A list of node IDs and image shifts.
    central_node_id
        The node ID of the central node.

    Returns
    -------
    A Ring object with the nodes in the correct positions.
    """

    # the first entry is the central node.
    ordered_node_list = get_ordered_node_list(node_list, central_node_id)

    # get the nodes in the correct positions.
    shifted_nodes = [
        topology[node_id].apply_image_shift(topology.lattice, shift) 
        for node_id, shift in ordered_node_list
    ]

    # the angle is defined for this central node by the neighbouring two nodes.
    angle = to_tuple([ordered_node_list[-1], ordered_node_list[1]])

    return Ring(nodes=shifted_nodes, angle=angle)


def ring_list_to_cluster(
    topology: Topology,
    ring_list: list[list[tuple[int, npt.NDArray[np.int_]]]],
    central_node_id: int
) -> Cluster:
    """ Convert a list of rings to a Cluster object.

    Parameters
    ----------
    topology
        The topology object containing the nodes.
    ring_list
        A list of lists of node IDs and image shifts.
    central_node_id
        The node ID of the central node.

    Returns
    -------
    A Cluster object with the rings in the correct positions.
    """

    rings = [node_list_to_ring(topology, x, central_node_id) for x in ring_list]
    return Cluster(central_node_id, rings)


def get_clusters(
    topology: Topology,
    all_ring_list: list[list[list[tuple[int, npt.NDArray[np.int_]]]]],
):
    """ Convert a list of lists of rings to a list of Cluster objects.

    Parameters
    ----------
    topology
        The topology object containing the nodes.
    all_ring_list
        A list of lists of lists of node IDs and image shifts.

    Returns
    -------
    A list of Cluster objects with the rings in the correct positions.
    """

    return [
        ring_list_to_cluster(topology, ring_list, central_node_id)
        for central_node_id, ring_list in enumerate(all_ring_list, start=1)
    ]


############################### ANALYSIS ###############################


def _get_single_vertex_symbol(cluster: Cluster) -> VertexSymbol:
    """
    Obtain the vertex symbol of a cluster, storing both the `all_rings=False` 
    and `all_rings=True` vectors.

    For 4-connected nets (with six angles), opposite angles are paired. If ring 
    sizes coincide, the 'all_rings' vector is used to break ties.

    Please note, while I have attempted to match the ordering scheme with 
    respect to ToposPro, the exact ordering may differ slightly in cases where
    the minimum ring sizes are identical at different angles but the 'all_rings'
    vector differs.

    TODO: I derive the angles from the rings found about the central node. This
    therefore will miss any angles that do not contain any rings, which needs
    correcting to match the ToposPro implementation (and indeed, provide the
    exact Vertex Symbol)!

    Parameters
    ----------
    cluster
        The cluster from which to obtain the vertex symbol.

    Returns
    -------
    The vertex symbol with both standard and all-rings representations.
    """

    # Calculate ring sizes with all_rings set to False and True.
    ring_sizes_by_angle = get_ring_size_by_angle(cluster, all_rings=False)
    all_ring_sizes_by_angle = get_ring_size_by_angle(cluster, all_rings=True)

    # Prepare the two vectors we wish to return.
    standard_vector = []
    all_rings_vector = []

    # If there are six angles (4-connected nets), pair opposite angles.
    if number_of_angles(cluster) == 6:
        pairs_data = []

        for angle_pair in get_opposite_angles(get_unique_angles(cluster)):

            # Sort the pair by ring sizes for each angle.
            pair_dict = {a: ring_sizes_by_angle[a] for a in angle_pair}
            sorted_pairs = sorted(pair_dict.items(), key=lambda x: tuple(x[1]))

            ring_sizes_tuple = tuple(x[1] for x in sorted_pairs)
            angles_tuple = tuple(x[0] for x in sorted_pairs)

            # If ring sizes are identical, sort by the 'all_rings' vector.
            if len({tuple(s) for s in ring_sizes_tuple}) == 1:
                all_pair_dict = {
                    a: all_ring_sizes_by_angle[a] for a in angle_pair
                }
                all_sorted_pairs = sorted(
                    all_pair_dict.items(), key=lambda x: tuple(x[1])
                )
                angles_tuple = tuple(x[0] for x in all_sorted_pairs)

            pairs_data.append((ring_sizes_tuple, angles_tuple))

        # Sort the collected data by ring sizes.
        pairs_data.sort(
            key=lambda data: tuple(i for sizes in data[0] for i in sizes)
        )

        # Extend our final vectors.
        for ring_sizes_tuple, angles_tuple in pairs_data:
            standard_vector.extend(ring_sizes_tuple)
            for angle in angles_tuple:
                all_rings_vector.append(all_ring_sizes_by_angle[angle])

    else:
        # If not a 4-connected net with six angles, just flatten the ring sizes.
        for sizes in ring_sizes_by_angle.values():
            standard_vector.append(sizes)

        for angle in ring_sizes_by_angle:
            all_rings_vector.append(all_ring_sizes_by_angle[angle])

    return VertexSymbol(
        vector=standard_vector,
        vector_all_rings=all_rings_vector,
    )


def get_vertex_symbol(
    clusters: Cluster | list[Cluster]
) -> VertexSymbol | list[VertexSymbol]:
    """
    Obtain the vertex symbol(s) for one or more clusters, storing both the 
    `all_rings=False` and `all_rings=True` vectors. If a list of clusters is
    provided, this function returns a list of vertex symbols in the same order.

    Parameters
    ----------
    clusters
        Either a single cluster, or a list of cluster objects from which to 
        obtain vertex symbols.

    Returns
    -------
    VertexSymbol or list of VertexSymbol
    - If a single Cluster is passed, returns a single VertexSymbol.
    - If a list of Cluster objects is passed, returns a list of VertexSymbols.
    """
    if isinstance(clusters, list):
        return [_get_single_vertex_symbol(cluster) for cluster in clusters]
    else:
        return _get_single_vertex_symbol(clusters)


def get_carvs_std_dev(carvs_vectors: npt.NDArray[np.int_]) -> float:
    """ Compute the standard deviation amongst the node environments. 
    
    Parameters
    ----------
    carvs_vectors
        The CARVS vectors for each cluster in the network.

    Returns
    -------
    The standard deviation of the CARVS vectors.
    """

    carvs_vectors = np.asarray(carvs_vectors, dtype=np.float64)

    if carvs_vectors.ndim == 1:
        return 0.0
    
    # ``average node environment``.
    carvs_mean = np.mean(carvs_vectors, axis=0) 

    # distance of each node from the average node environment.
    carvs_dist_from_mean = np.linalg.norm(carvs_vectors - carvs_mean, axis=1)

    # spread of node environments.
    carvs_std_dev = np.sqrt(
        np.square(carvs_dist_from_mean).sum() / carvs_dist_from_mean.shape[0]
    )
    
    return float(carvs_std_dev)


def get_carvs_vector(
    cluster: list[Cluster] | Cluster,
    max_size: int | None = None,
    return_per_cluster: bool = False,
    unique: bool = False,
) -> CARVS | npt.NDArray[np.floating]:
    """
    Get the Cummulative All-Rings Vertex Symbol (CARVS) vector.

    If the input is a ``Cluster`` object, the CARVS is calculated for that
    specific cluster. If the input is a list of ``Cluster`` objects, the CARVS 
    vector is calculated as an average over all ``clusters`` in the entire 
    network, by calling the ``get_carvs_vector`` method for each cluster.

    Parameters
    ----------
    cluster
        The input topology object (either a ``Cluster`` or a list of ``Cluster``
        objects).
    max_size
        The maximum ring size to consider. If not specified, the maximum ring
        size in the network is used.
    return_per_cluster
        Whether to return the CARVS vector for each cluster in the network, or 
        to return the average CARVS vector over all clusters.
    unique
        If returning the CARVS vector for each cluster, whether to return only
        unique CARVS vectors.

    Returns
    -------
    The CARVS vector.
    """

    if isinstance(cluster, Cluster):

        size, counts = np.unique(get_ring_sizes(cluster), return_counts=True)
        max_size = max(size) if max_size is None else max_size
    
        carvs = np.zeros(shape=(max_size,), dtype=np.int_)
        carvs[size-1] = counts
    
        return CARVS(vector=carvs, spread=0.0, is_single_node=True)

    elif isinstance(cluster, list):

        found_max_size = max([largest_ring_size(c) for c in cluster])

        if max_size is None:
            max_size = found_max_size

        elif max_size is not None and found_max_size > max_size:
            max_size = found_max_size
            warnings.warn(
                f"The maximum ring size in the network is {max_size}, "
                f"which is greater than the specified maximum size. "
                f"Setting the maximum size to {max_size}.",
                UserWarning,
                stacklevel=2
            )

        carvs = np.zeros(shape=(len(cluster), max_size))
        for i, c in enumerate(cluster):
            carvs[i] = get_carvs_vector(c, max_size=max_size).vector

        if return_per_cluster:
            carvs = carvs.astype(np.int_)

            if unique:
                return np.unique(carvs, axis=0)
            
            return carvs

    return CARVS(
        vector=np.mean(carvs, axis=0),
        spread=get_carvs_std_dev(carvs),
        is_single_node=False
    )