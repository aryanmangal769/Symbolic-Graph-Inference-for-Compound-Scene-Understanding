import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

class Connectivity_head(nn.Module):

    def __init__(self,feature_dim):
        super().__init__()
        self.feature_dim = feature_dim
        self.linear = nn.Linear(feature_dim, feature_dim)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()


    def forward(self, h , A):
        """
        Computing the importance of each node in the graph
        """
        # print(h)
        h = self.linear(h)
        h = self.relu(h)
        h = F.normalize(h, p=1, dim=1)
        h = A@h
        h = torch.sum(h,1)
        return h
