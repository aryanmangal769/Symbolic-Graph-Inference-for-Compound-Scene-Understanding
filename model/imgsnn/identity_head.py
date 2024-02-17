import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

class Identity_head(nn.Module):

    def __init__(self,feature_dim, img_cond = False, input_dim = 100, num_classes = 10):
        super().__init__()
        self.feature_dim = feature_dim
        self.linear = nn.Linear(feature_dim, feature_dim)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.img_cond = img_cond

        if self.img_cond:
            self.linear_img = nn.Linear(feature_dim+ input_dim, num_classes)
            # self.linear_img = nn.Linear(input_dim, num_classes)
        else:
            self.linear = nn.Linear(feature_dim, num_classes)

    def forward(self, h , img_feat = None):
        """
        Computing the importance of each node in the graph
        """

        if self.img_cond:
            h = torch.sum(h,0)
            img_feat = torch.cat((h.unsqueeze(0),img_feat),1)
            h = self.linear_img(img_feat)
        
        else:
            h = torch.sum(h,0)
            h = self.linear(h)

        return h
