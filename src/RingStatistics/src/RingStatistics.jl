module RingStatistics

    using PyCall, Graphs, PeriodicGraphs, PeriodicGraphEmbeddings, CrystalNets

    export greet
    export run_bfgs
    export get_topology
    export run_rings, run_strong_rings
    export run_coordination_sequences


    """
        import_check()
    """
    function import_check()
        println("RingStatistics is accessible!")
    end

    """
        run_bfgs(edges, v, depth)

    nv(g::PeriodicGraph3D)

    Calculate the number of nodes in the given periodic graph.
    """
    function nv(g::PeriodicGraph3D)
        length(g.nlist)
    end

    """
        ne(g::PeriodicGraph3D)

    Calculate the number of edges in the given periodic graph.
    """
    function ne(g::PeriodicGraph3D)
        g.ne[]
    end

    """
        graph_from_edges(edges)

    Construct a periodic graph from a given set of edges.
    """
    function graph_from_edges(edges)
        pedges = [
            PeriodicEdge3D(edge[1], edge[2], (edge[3],edge[4],edge[5])) 
            for edge in eachrow(edges)
        ]
        PeriodicGraph3D(pedges)
    end

    """
    count_rings(graph, depth)

    Count the number of rings up to a specified depth in the given graph.
    If a ring larger than `depth` is found, a warning is issued (once per ring 
    size) and the ring is ignored. Returns a two-dimensional array where the 
    first column is the ring size and the second column is the ring count.
    """
    function count_rings(graph, depth)
        rcount = zeros(Int, 2*depth+3)
        warned_sizes = Set()
        for r in sort!(first(graph))
            ring_size = length(r)
            if ring_size > 2*depth+3
                if ring_size ∉ warned_sizes
                    @warn "Ring size $ring_size exceeds the specified depth $depth. This ring will be ignored."
                    push!(warned_sizes, ring_size)
                end
                continue
            end
            rcount[ring_size] += 1
        end
        return hcat(1:length(rcount), rcount)
    end

    """
        gather_ring_nodes(graph)

    Gather the ring nodes for each node in the graph. Returns two lists:
    1. The number of rings each node is a part of.
    2. The rings associated with each node.
    """
    function gather_ring_nodes(graph, depth)
        ras = RingAttributions(graph, false, depth)
        num_rings_per_node = length.(ras.attrs)
        rings_per_node = [collect_rings(node) for node in ras]
        num_rings_per_node, rings_per_node
    end

    """
    gather_ring_nodes(graph)

    Gather the ring nodes for each node in the graph. Returns two lists:
    1. The number of rings each node is a part of.
    2. The rings associated with each node.
    """
    function gather_strong_ring_nodes(graph, depth)
        ras = RingAttributions(graph, true, depth)
        num_rings_per_node = length.(ras.attrs)
        rings_per_node = [collect_rings(node) for node in ras]
        num_rings_per_node, rings_per_node
    end

    """
        collect_rings(node)

    Helper function to collect rings for a given node.
    """
    function collect_rings(node)
        [collect_ring(ring) for ring in node]
    end

    """
        collect_ring(ring)

    Helper function to collect vertices for a given ring.
    """
    function collect_ring(ring)
        [(vertex.v, vertex.ofs) for vertex in ring]
    end

    """
        run_rings(edges)

    Main function to run the ring statistics computation. Takes in the edges,
    constructs a graph, and computes ring statistics and ring nodes.
    """
    function run_rings(edges, depth=12)
        g = graph_from_edges(edges)
        rcount = count_rings(rings(g, depth), depth)
        nrnodes, rnodes = gather_ring_nodes(g, depth)
        rcount, nrnodes, rnodes
    end

    """
        run_strong_rings(edges)

    Main function to run the ring statistics computation. Takes in the edges,
    constructs a graph, and computes ring statistics and ring nodes.
    """
    function run_strong_rings(edges, depth=12)
        g = graph_from_edges(edges)
        rcount = count_rings(strong_rings(g, depth), depth)
        nrnodes, rnodes = gather_strong_ring_nodes(g, depth)
        rcount, nrnodes, rnodes
    end

    function run_coordination_sequences(edges, dmax)
        g = graph_from_edges(edges)
        num_vertices = nv(g)
        coord_sequences = []
        for v in 1:num_vertices
            seq = coordination_sequence(g, v, dmax)
            push!(coord_sequences, seq)
        end
        return coord_sequences
    end

    function run_bfs(edges, v::Int, depth::Int)
        g = graph_from_edges(edges)
        return node_indices_and_offsets(g, v, depth)
    end

    """
        node_indices_and_offsets(g::PeriodicGraph, v::Integer, dmax)

    For a given vertex `v` in a periodic graph `g`, compute the node indices
    and offsets for nodes at each distance up to `dmax`.

    Returns a vector where each entry corresponds to distance `d`, and contains
    a list of tuples `(node_index, offset_vector)` for nodes at that distance.
    """
    function node_indices_and_offsets(g::PeriodicGraph, v::Integer, dmax)
        Q = Graphs._neighborhood(g, v, dmax, weights(g), outneighbors)
        popfirst!(Q)
        result = [Vector{Tuple{Int, Tuple{Int, Int, Int}}}() for _ in 1:dmax]
        for (node, distance) in Q
            if distance <= dmax
                offset_tuple = Tuple(node.ofs)
                push!(result[distance], (node.v, offset_tuple))
            end
        end
        return result
    end

    """ 
    get_topological_genome(nodes, edges, cell)

    Construct a periodic graph embedding from a given set of nodes, edges, and 
    cell, and then determine the topology.
    """
    function get_topological_genome(
        nodes,         # N×3 array
        edges,         # M×5 array
        cell_lengths,  # length-3 vector (a, b, c)
        cell_angles    # length-3 vector (α, β, γ)
    )

        CrystalNets.toggle_export(false); 
        CrystalNets.toggle_warning(false);
        
        cell_lengths = collect(cell_lengths)
        cell_angles = collect(cell_angles)
    
        @assert size(nodes,1) == 3 "Nodes must have 3 coordinates per point."
        @assert size(edges,2) == 5 "Each edge should be a row with 5 entries."
        @assert length(cell_lengths) == 3 "Lengths must be a 3-element vector."
        @assert length(cell_angles) == 3 "Angles must be a 3-element vector."
    
        graph = graph_from_edges(edges)
        cell = Cell(1, cell_lengths, cell_angles)
        embedding = PeriodicGraphEmbedding3D(graph, nodes, cell)
        topology, _ = only(topological_genome(embedding.g))

        return string(topology)

    end

end # module RingStatistics
