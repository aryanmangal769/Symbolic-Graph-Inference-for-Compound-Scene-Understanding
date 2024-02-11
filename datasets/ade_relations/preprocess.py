import cv2
import os
from torch.utils.data import Dataset
import torch
import sys
from tqdm import tqdm
sys.path.append('.')
sys.path.append('../..')
import pickle
import ijson
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import utils_ade20k
from collections import Counter
import cv2
import pickle as pkl
from tqdm import tqdm
from model.build_model import  build_vit
import yaml


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

obj_feat = 128
pos_feat = 16

config_file = '../../configs/train_ade.yaml'
f = open(config_file, 'r', encoding='utf-8')
configs = f.read()
configs = yaml.safe_load(configs)
configs['num_gpu'] = [2]
configs['vit']['num_classes'] = obj_feat
configs['vit']['head'] = False

model_vit = build_vit(configs)
model_vit.eval()

DATASET_PATH = '/data/aryan/Seekg/Datasets/ade20k/ADE20K_2021_17_01/'
index_file = 'index_ade20k.pkl'
with open('{}/{}'.format(DATASET_PATH, index_file), 'rb') as f:
    index_ade20k = pkl.load(f)

relation_file = '../ade20k/annotated_images/image_object_relations.pkl'
with open(relation_file, 'rb') as f:
    image_object_relations = pkl.load(f)

preprocesses_relations = []

for file_name, relations in image_object_relations.items():
    print("######")
    if len(relations) == 0:
        continue
    index = index_ade20k['filename'].index(file_name)
    full_file_name = '{}/{}'.format(index_ade20k['folder'][index], index_ade20k['filename'][index])


    root_path = DATASET_PATH.replace('ADE20K_2021_17_01/', '')
    info = utils_ade20k.loadAde20K('{}/{}'.format(root_path, full_file_name))

    img = Image.open(info['img_name'])

    # Extract object information
    all_objects = info['objects']['class']
    all_polygons = info['objects']['polygon']

    # Filter polygons based on area
    min_area_threshold = 15000  
    max_area  = img.size[0] * img.size[1]
    filtered_objects = []
    filtered_polygons = []

    for obj, poly in zip(all_objects, all_polygons):
        area = cv2.contourArea(np.array(list(zip(poly['x'], poly['y']))))
        if area > min_area_threshold and area < max_area/3:
            filtered_objects.append(obj)
            filtered_polygons.append(poly)

    element_counts = {}
    converted_list = []
    for element in filtered_objects:
        if element not in element_counts:
            element_counts[element] = 0
        converted_list.append(f"{element}_{element_counts[element]}")
        element_counts[element] += 1
    
    filtered_objects = converted_list


    for relation in relations:
        objects = relation['objects']
        relation = relation['relation']
        
        relation_embed = {}       
        relation_embed['relation'] = relation
        # relation_embed['img'] = model_vit(transform(img).unsqueeze(0)).cpu().detach()
        
        for i, obj in enumerate(objects):
            # Index the object from filtered_objects
            obj_index = filtered_objects.index(obj)
            
            # Get the corresponding polygon
            obj_polygon = filtered_polygons[obj_index]
            
            # Convert polygon to bounding box
            x_min = min(obj_polygon['x'])
            y_min = min(obj_polygon['y'])
            x_max = max(obj_polygon['x'])
            y_max = max(obj_polygon['y'])
            
            # Extract object region from the image
            object_region = img.crop((x_min, y_min, x_max, y_max))
            
            # Apply transformations
            transformed_object = transform(object_region)

            img_feat = model_vit(transformed_object.unsqueeze(0))
            bbox_feat = torch.tensor([x_min/img.size[0], y_min/img.size[1], x_max/img.size[0], y_max/img.size[1]])
            # print(img_feat.shape, bbox_feat.shape)

            concatenated_feat = torch.cat((img_feat, bbox_feat.unsqueeze(0).to(img_feat.device)), dim=1)
            # print(concatenated_feat.shape)

            relation_embed['object'+str(i)] = concatenated_feat.cpu().detach()
        
        preprocesses_relations.append(relation_embed)

with open('preprocessed_relations.pkl', 'wb') as f:
    pkl.dump(preprocesses_relations, f)






