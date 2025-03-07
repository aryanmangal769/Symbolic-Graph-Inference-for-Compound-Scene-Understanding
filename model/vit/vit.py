import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm


class ViT(nn.Module):
    def __init__(self, num_classes=10, head=True):
        super(ViT, self).__init__()
        self.model = timm.create_model('vit_base_patch16_224', pretrained=True)
        if head == True:
            self.model.head = nn.Linear(self.model.head.in_features, num_classes)
        else :
            self.model.head = nn.Identity()

    def forward(self, x):
        return self.model(x)