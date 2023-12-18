import cv2
import os
from torch.utils.data import Dataset
import torch
import sys
from tqdm import tqdm
sys.path.append('.')
import pickle
import ijson

from utils.dataset_utils import get_bbox_from_segment

class ADE_20k(Dataset):
    def __init__(self,
                base_dir : str,
                subset_path : str,
                split : str = 'train',
                split_size : int = 0.8
                ):
        super().__init__()
        self.base_dir = base_dir
        self._load_dataset()

    def _load_dataset(self):
        self.paths = []
        self.objects = []
        self.verbs = []
        self.bboxes = []

        with open(self.base_dir, 'rb') as f:
            image_info_dict = pickle.load(f)
        
        for image_path in image_info_dict.keys():
            self.paths.append(image_path)
            self.verbs.append(image_info_dict[image_path]['scene_name'])
            self.objects.append(image_info_dict[image_path]['most_common_object_name'])
            self.bboxes.append(self.format_bbox(image_info_dict[image_path]['object_names'],image_info_dict[image_path]['object_bboxes']))
  

    def format_bbox(self, object_list, bbox):
       
        element_counts = {}

        converted_list = []
        for element in object_list:
            if element not in element_counts:
                element_counts[element] = 0
            converted_list.append(f"{element}_{element_counts[element]}")
            element_counts[element] += 1
        
        bbox_formated = []
        for i, element in enumerate(converted_list):
            bbox_formated.append({'object': element, 'object_bbox': bbox[i]})  
        return bbox_formated

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        # print(path)
        obj = self.objects[idx]
        verb = self.verbs[idx]
        bbox = self.bboxes[idx]
        return  [bbox,(obj, verb)]

        