import cv2
import os
from torch.utils.data import Dataset
import torch
import sys
from tqdm import tqdm
sys.path.append('.')
import pickle
import ijson
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import cv2
import pickle as pkl
from tqdm import tqdm


class ADE_relations(Dataset):
    def __init__(self,
                data_file : str == "/data/aryan/Seekg/MGNN/datasets/ade_relations/preprocessed_relations.pkl"
                ):
        super().__init__()
        self.data_file = data_file
        self.spatial_relations = ["on", "next to", "behind", "in front of",  "above", "across", "below", "inside", "under", "left" , "right", "in" , "None"]
        self._load_dataset()

    def _load_dataset(self):
        with open(self.data_file, 'rb') as f:
            self.object_relations = pickle.load(f)
        
        self.relations = []
        self.embeddings = []
        for relation in self.object_relations:
            self.relations.append(self.spatial_relations.index(relation['relation']))
            # print(self.spatial_relations.index(relation['relation']))
            # self.embeddings.append(torch.cat([torch.tensor(relation['img']), torch.tensor(relation['object0']), torch.tensor(relation['object1'])], dim=1).squeeze(0)  )
            self.embeddings.append(torch.cat([ torch.tensor(relation['object0']), torch.tensor(relation['object1'])], dim=1).squeeze(0)  )

    def __len__(self):
        return len(self.relations)

    def __getitem__(self, idx):
        return self.relations[idx], self.embeddings[idx]
