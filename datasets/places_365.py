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

class PLACES_365(Dataset):
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

        for category in os.listdir(self.base_dir):
            category_dir = os.path.join(self.base_dir, category)

            with open(os.path.join(category_dir), 'rb') as f:
                category_bboxes_labels = pickle.load(f)

            for image_path in category_bboxes_labels.keys():
                self.paths.append(image_path)
                self.bboxes.append(format_bbox(category_bboxes_labels[image_path]))
                self.verbs.append(category)

                category = category.split('_')[0]

                if category == 'aquarium':
                    self.objects.append('bird')
                elif category == 'airfield':
                    self.objects.append('army_base')
                elif category == 'army_base':
                    self.objects.append('person')
                elif category == 'restaurant':
                    self.objects.append('dining table')
                elif category == 'gas_station':
                    self.objects.append('car')
                elif category == 'classroom':
                    self.objects.append('chair')
                elif category == 'computer_room':
                    self.objects.append('tv')
                elif category == 'hospital_room':
                    self.objects.append('bed')
                elif category == 'waterfall':
                    self.objects.append('cat')
                elif category == 'kitchen':
                    self.objects.append('oven')     
                elif category == 'parking_lot':
                    self.objects.append('car')         
                elif category == 'ballroom':
                    self.objects.append('person')
                elif category == 'beach':
                    self.objects.append('person')
                elif category == 'bedroom':
                    self.objects.append('bed')

                elif category == 'botanical_garden':
                    self.objects.append('potted plant')
                elif category == 'food_court':
                    self.objects.append('chair')     

    def format_bbox(self, bbox):
        bbox_formated = []
        for obj in bbox:
            for i, obj in enumerate(bbox[obj]):
                bbox_formated.append({'object': obj['label']+'_'+i, 'object_bbox': obj['bbox']})  
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

        