import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

class Fusion_net(nn.Module):

    def __init__(self,feature_dim):
        super().__init__()
        self.feature_dim = feature_dim
        self.linear = nn.Linear(2*feature_dim, feature_dim)
        self.softmax = nn.Softmax(dim=0)


    def forward(self, conn_feat , id_feat):
        """
        Computing the importance of each node in the graph
        """

        # pdb.set_trace()
        merged_feat = torch.cat((conn_feat,id_feat),0)
        h = self.linear(merged_feat)
        h = self.softmax(h)

        return h