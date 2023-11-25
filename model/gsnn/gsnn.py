import torch
import torch.nn as nn
from model.graph_models.gcn import GCN
from model.gsnn.importance_net import Importance_net
from utils.graph_utils import get_neighbour_nodes
import torch.nn.functional as F



class GSNN(nn.Module):
    '''
    GSNN: Graph Search Neural Network
    '''

    def __init__(self, config):
        super().__init__()
        self.config = config

        nfeat = config['gcn']['n_feat']
        nhid = config['gcn']['n_hid']
        nout = config['gcn']['n_out']
        dropout = config['gcn']['dropout']
        n_layers = config['gcn']['n_layers']

        self.gcn = GCN(nfeat, nhid, nout,n_layers, dropout)
        self.imp = Importance_net(nout)

    def forward(self, x, adj, active_idx):
        '''
        x: Knowledge graph embedding
        adj: Adjacency matrix of the knowledge graph
        active_idx: Active node index
        '''
        neighbor_idx = get_neighbour_nodes(adj, active_idx)

        current_idx = torch.cat(( active_idx, neighbor_idx)) 
        h = self.gcn(x[current_idx, :].clone(), adj[current_idx][:, current_idx].clone())
        imp = self.imp(h, adj[current_idx][:,current_idx])

        imp = F.normalize(imp, p=1, dim=0)
        values, indices = torch.topk(imp[len(active_idx):], k=3)
        # print(values.requires_grad)

        # return values, neighbor_idx[indices].float().requires_grad_(True)
        return imp[len(active_idx):], neighbor_idx

