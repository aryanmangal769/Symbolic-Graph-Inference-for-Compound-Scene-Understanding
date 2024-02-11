import cv2
from PIL import Image
import numpy as np
import sys
sys.path.append('.')
sys.path.append('datasets/ade20k')
import utils_ade20k
import torch
from torchvision import transforms
from model.build_model import build_vit
import pickle as pkl
import yaml
from model.relation_net.relation_net import RelationNet

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def extract_object_info(image_path):
    # Load the image
    img = Image.open(image_path)
    
    # Extract object information
    all_objects = utils_ade20k.loadAde20K(image_path)['objects']['class']
    all_polygons = utils_ade20k.loadAde20K(image_path)['objects']['polygon']
    
    # Filter polygons based on area
    min_area_threshold = 15000
    max_area = img.size[0] * img.size[1]
    filtered_objects = []
    filtered_polygons = []

    for obj, poly in zip(all_objects, all_polygons):
        area = cv2.contourArea(np.array(list(zip(poly['x'], poly['y']))))
        if area > min_area_threshold and area < max_area / 3:
            filtered_objects.append(obj)
            filtered_polygons.append(poly)
    
    # Convert polygons to bounding boxes
    bounding_boxes = []
    for poly in filtered_polygons:
        x_min = min(poly['x'])
        y_min = min(poly['y'])
        x_max = max(poly['x'])
        y_max = max(poly['y'])
        bounding_boxes.append((x_min, y_min, x_max, y_max))
    
    return filtered_objects, bounding_boxes

def preprocess_image(image_path, model_vit):
    # Extract object information
    objects, bounding_boxes = extract_object_info(image_path)
    
    # Initialize list to store features
    object_features = []
    
    # Load the image
    img = Image.open(image_path)
    
    for bbox in bounding_boxes:
        # Extract object region from the image
        object_region = img.crop(bbox)
        
        # Apply transformations
        transformed_object = transform(object_region)
        
        # Pass through ViT model
        img_feat = model_vit(transformed_object.unsqueeze(0))
        
        # Convert bounding box to tensor and concatenate with image feature
        bbox_feat = torch.tensor([bbox[0] / img.size[0], bbox[1] / img.size[1], bbox[2] / img.size[0], bbox[3] / img.size[1]])
        concatenated_feat = torch.cat((img_feat, bbox_feat.unsqueeze(0).to(img_feat.device)), dim=1)
        
        object_features.append(concatenated_feat.cpu().detach())
    
    return objects, object_features

def predict_relation(object_features, relation_model):
    with torch.no_grad():
        outputs = relation_model(object_features)
        print(outputs)
        _, predicted = torch.max(outputs, 1)
        predicted_relation = predicted.item()
    return predicted_relation

# Example usage:
config_file = 'configs/train_ade.yaml'
f = open(config_file, 'r', encoding='utf-8')
configs = f.read()
configs = yaml.safe_load(configs)
configs['num_gpu'] = [2]
configs['vit']['num_classes'] = 128  # Assuming this is the correct value
configs['vit']['head'] = False

model_vit = build_vit(configs)
model_vit.eval()

config_file = 'configs/train_relationnet.yaml'
f = open(config_file, 'r', encoding='utf-8')
configs = f.read()
configs = yaml.safe_load(configs)

relation_model = RelationNet(configs)  # Assuming this is the correct initialization method for your RelationNet model
relation_model.load_state_dict(torch.load("trained_model.pth"))  # Assuming you have a pre-trained RelationNet model
relation_model.eval()

image_path = '/data/aryan/Seekg/Datasets/ade20k/ADE20K_2021_17_01/images/ADE/training/home_or_hotel/bedroom/ADE_train_00000355.jpg'  # Update with the path to your image
objects, object_features = preprocess_image(image_path, model_vit)
print("Objects:", objects)
# print(object_features[8])
feat = torch.cat([object_features[8], object_features[3]], dim=1).float()
predicted_relation = predict_relation(feat, relation_model)
print("Predicted relation:", predicted_relation)
