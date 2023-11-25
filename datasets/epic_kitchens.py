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

class EPIC_Kitchens(Dataset):
    def __init__(self,
                base_dir : str,
                subset_path : str,
                split : str = 'train',
                split_size : int = 0.8
                ):
        super().__init__()
        self.base_dir = base_dir
        if subset_path is not None:
            self.subset_path = subset_path
            self.subset = pickle.load(open(subset_path, 'rb'))
        self.split = split
        self.split_size = split_size

        self.img_dir = os.path.join(self.base_dir, "EPIC_Aff_images")
        self.label_dir = os.path.join(self.base_dir, 'EPIC_Aff_50_classes_2d_output_labels')
        self.rgb = 'selected_plus_guided_rgb'
        self.label_2d = '50_classes_2d_output_labels'

        if subset_path is not None: 
            self._load_dataset()

        self.VISOR_path = os.path.join(self.base_dir, 'VISOR_annotations')
        self.VISOR_json_dir_dense = os.path.join(self.VISOR_path, 'Interpolations-DenseAnnotations', 'train')
        self.VISOR_json_dir_sparse = os.path.join(self.VISOR_path, 'GroundTruth-SparseAnnotations', 'annotations', 'train')
        self._load_VISOR_annotations()

    def _load_VISOR_annotations(self):
        self.all_dense_VISOR_jsons = {}
        self.all_sparse_VISOR_jsons = {}
        for root, dirs, files in os.walk(self.VISOR_json_dir_sparse):
            for file in files:
                kitchen_name = file.split('_')[0]
                sequence_name = file.split('_')[1].split('.')[0]
                if file.endswith('.json'):
                    dense_file = os.path.join(self.VISOR_json_dir_dense, file.split('.')[0] + '_interpolations'+ '.json')
                    sparse_file = os.path.join(self.VISOR_json_dir_sparse, file)
                    self.all_sparse_VISOR_jsons[kitchen_name + '_' + sequence_name] =  sparse_file
                    self.all_dense_VISOR_jsons[kitchen_name + '_' + sequence_name] =  dense_file

    def _load_dataset(self):
        self.paths = []
        self.objects = []
        self.verbs = []

        for obj in self.subset:
            for verb in self.subset[obj]:
                for path in self.subset[obj][verb]:
                    self.paths.append(path)
                    self.objects.append(obj)
                    self.verbs.append(verb)

    def VISOR_bbox(self, img_name, sequence):
        VISOR_active_objects_list = []
        VISOR_bboxs = []
        VISOR_active_objects, divisor = self.read_VISOR_annot(img_name, sequence)

        for e_idx, entity in enumerate(VISOR_active_objects): 
            VISOR_active_objects_list.append(entity['name']) 
            bbox = get_bbox_from_segment(entity['segments'])
            VISOR_bboxs.append({'object': entity['name'], 'object_bbox': tuple([int(item / divisor) for item in bbox])})
        return VISOR_bboxs, VISOR_active_objects_list


    def read_VISOR_annot(self, img_name, sequence):
        VISOR_filename = self.all_sparse_VISOR_jsons[sequence]
        the_annotation = None
        with open(VISOR_filename, 'r') as f:
            # print(json.load(f))
            VISOR_annot = ijson.items(f, 'video_annotations.item')
            # print(VISOR_annot)
            for entity in VISOR_annot:
                if entity['image']['name'].split('.')[0] == img_name.split('.')[0]:
                    # print(entity)
                    the_annotation = entity['annotations']
                    divisor = 2.25
                    break
        if the_annotation is None:
            print("No sparse annotation for ", img_name)
            VISOR_filename = self.all_dense_VISOR_jsons[sequence]
            with open(VISOR_filename, 'r') as f:
                VISOR_annot = ijson.items(f, 'video_annotations.item')
                for entity in VISOR_annot:
                    if entity['image']['name'].split('.')[0] == img_name.split('.')[0]:
                        the_annotation = entity['annotations']
                        divisor = 1
                        break
        return the_annotation, divisor

    
    def has_sparse_annotations(self, img_name, sequence):
        if sequence not in self.all_sparse_VISOR_jsons:
            return False
        VISOR_filename = self.all_sparse_VISOR_jsons[sequence]
        with open(VISOR_filename, 'r') as f:
            VISOR_annot = ijson.items(f, 'video_annotations.item')
            for entity in VISOR_annot:
                if entity['image']['name'].split('.')[0] == img_name.split('.')[0]:
                    return True
        return False

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        # print(path)
        obj = self.objects[idx]
        verb = self.verbs[idx]

        kitchen, sample_id = path.split('/')
        img_path = os.path.join(self.img_dir, kitchen, self.rgb, sample_id + '.jpg')  
        # print(img_path)      
        label_2d_path = os.path.join(self.label_dir, kitchen, self.label_2d, sample_id + '.pkl')
        
        sequence = sample_id.split('_')[0] + '_' + sample_id.split('_')[1]
        VISOR_bboxs, _ = self.VISOR_bbox(sample_id + '.jpg', sequence)

        return  [VISOR_bboxs,(obj, verb)]

        