import torch
import torch.nn as nn
from model.graph_models.gcn import GCN
from model.gsnn.importance_net import Importance_net
from utils.graph_utils import get_neighbour_nodes
from model.gsnn.gsnn import GSNN

class MGNN(nn.Module):
    '''
    MGNN: Multi Graph Neural Network
    '''

    def __init__(self, config):
        super().__init__()
        self.config = config

        nfeat = config['gcn']['n_feat']
        nhid = config['gcn']['n_hid']
        nout = config['gcn']['n_out']
        dropout = config['gcn']['dropout']
        n_layers = config['gcn']['n_layers']

        self.GSNN = GSNN(config)
    
    def forward(self, ):