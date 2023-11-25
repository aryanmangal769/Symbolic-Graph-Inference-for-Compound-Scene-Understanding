import torch

def get_neighbour_nodes(A, active_nodes):
    connected_cols = torch.where(A[:, active_nodes].any(axis=1))[0]
    connected_nodes = torch.unique(connected_cols)
    connected_nodes = connected_nodes[~torch.eq(connected_nodes.unsqueeze(1), active_nodes).any(axis=1)]
    return connected_nodes
