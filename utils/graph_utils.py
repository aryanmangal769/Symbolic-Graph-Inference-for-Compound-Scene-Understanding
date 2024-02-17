import torch
import pdb

def get_neighbour_nodes(A, active_nodes):
    connected_cols = torch.where(A[:, active_nodes].any(axis=1))[0]
    connected_nodes = torch.unique(connected_cols)
    connected_nodes = connected_nodes[~torch.eq(connected_nodes.unsqueeze(1), active_nodes).any(axis=1)]
    return connected_nodes

def merge_graphs(x, graph1, graph2, x2, active_idx_init):
    # Combine nodes and create a new matrix
    combined_nodes = graph1.shape[0] + graph2.shape[0]
    combined_matrix = torch.zeros((combined_nodes, combined_nodes), dtype=graph1.dtype).to(x.device)

    # Copy adjacency matrices of the original graphs
    combined_matrix[:graph1.shape[0], :graph1.shape[1]] = graph1
    combined_matrix[graph1.shape[0]:, graph1.shape[1]:] = graph2

    # Connect specified nodes
    for i,edge in enumerate(active_idx_init[0]):
        # pdb.set_trace()
        combined_matrix[edge, graph1.shape[1] + active_idx_init[1][i]] = 1
        combined_matrix[graph1.shape[0] + active_idx_init[1][i], edge] = 1

    # Copy node embeddings
    combined_x = torch.rand((combined_nodes, x.shape[1])).to(x.device)
    combined_x[:x.shape[0], :] = x

    # for i,node in enumerate(active_idx_init[0]):
    #     combined_x[node, :] = x2[active_idx_init[1][i], :]

    return combined_x, combined_matrix