import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
import sys
sys.path.append('.')
import argparse 
import yaml
import pickle
from tqdm import tqdm

from model.build_model import build_gsnn 
from model.build_model import build_mgsnn
from model.mgnn.mgnn_loss import MGNNLoss
from datasets.epic_kitchens import EPIC_Kitchens
from utils.dataset_utils import custom_collate
from utils.scene_graph_utils import generate_SG_from_bboxs, get_KG_active_idx, visualize_graph, update_SG_adj, get_SG_active_idx
from utils.vis_utils import visualize_bbox
import numpy as np

import torch
import torchvision.transforms as T
import torch.nn.functional as F
from datasets.ade20k import ADE_20k
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from PIL import Image
import os
from collections import defaultdict, Counter
from heapq import nlargest
from tqdm import tqdm
import pickle
import pdb


def get_actions(idx, KG_path ):
    actions = []
    with open(KG_path, 'rb') as f:
        KG_embeddings, KG_adjacency_matrix, KG_vocab, KG_nodes = pickle.load(f)
    # print(len(KG_vocab))
    verbs = []
    for i in idx:
        print(KG_vocab[i])
        # print(KG_nodes['affordances'])
        if KG_vocab[i] in KG_nodes['affordances'][0]:
            verbs.append(KG_vocab[i])   
    for verb in verbs:
        nouns = []
        nouns_total = np.nonzero(KG_adjacency_matrix[KG_vocab.index(verb)])
        for noun in nouns_total:

            if noun.int() in idx:
                nouns.append(noun)

        actions.extend([[verb +" "+ KG_vocab[i]] for i in nouns])
        
    return actions

def get_accuracy(predictions, targets, threshold=0.5):
    """
    Compute accuracy given predicted outputs and ground truth labels.

    Args:
    - predictions: Tensor containing predicted outputs (e.g., model outputs)
    - targets: Tensor containing ground truth labels
    - threshold: Threshold for considering predictions as positive (default is 0.5)

    Returns:
    - accuracy: Accuracy value (percentage of correct predictions)
    """
    with torch.no_grad():
        # Compare any prediction with ground truth labels
        correct_predictions = any((pred == targets).all() for pred in predictions)

        # Calculate accuracy as the percentage of correct predictions
        accuracy = correct_predictions * 100.0

    return accuracy

def get_precision_recall(predictions, targets):
    """
    Compute precision and recall given predicted outputs and ground truth labels.

    Args:
    - predictions: List of predicted strings
    - targets: List of ground truth strings

    Returns:
    - precision: Precision value
    - recall: Recall value
    """
    with torch.no_grad():
        true_positives = sum(pred == target for pred, target in zip(predictions, targets))
        false_positives = sum(pred != target for pred, target in zip(predictions, targets))
        false_negatives = sum(pred != target for pred, target in zip(predictions, targets))

        precision = true_positives / (true_positives + false_positives + 1e-7)
        recall = true_positives / (true_positives + false_negatives + 1e-7)

    return precision, recall

CLASSES = ['__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
# Load the pre-trained Faster R-CNN model
model = fasterrcnn_resnet50_fpn_v2(pretrained=True)
model.eval()

def get_bboxes_labels(image_path, min_objects=5):
    # Load the image
    img = Image.open(image_path).convert("RGB")

    # Define the transformation
    transform = T.Compose([T.ToTensor()])

    # Apply the transformation
    img_tensor = transform(img)
    img_tensor = img_tensor.unsqueeze(0)  # Add a batch dimension

    # Make prediction
    with torch.no_grad():
        prediction = model(img_tensor)

    # Get bounding boxes, labels, and scores
    boxes = prediction[0]['boxes'].cpu().numpy()
    labels = prediction[0]['labels'].cpu().numpy()
    scores = prediction[0]['scores'].cpu().numpy()

    # Track unique object labels and their corresponding bounding boxes
    category_bboxes_labels = defaultdict(list)

    for box, label, score in zip(boxes, labels, scores):
        if score < 0.3:
            continue

        # Add bounding box and label to the category
        category_bboxes_labels[CLASSES[label]].append({
            'bbox': box,
            'label': CLASSES[label],
            'score': score
        })

    return category_bboxes_labels

def train(configs):
    dataset_name = configs['dataset']
    base_dir = configs['base_dir']
    subset_path = configs['subset_path']
    epochs = configs['epochs']
    batch_size = configs['batch_size']
    alpha = configs['alpha']
    lr = configs['lr']

   
    KG_path = configs['KG_path']    
    with open(KG_path, 'rb') as f:
        KG_embeddings, KG_adjacency_matrix, KG_vocab, KG_nodes = pickle.load(f)
    
    KG_embeddings = F.normalize(KG_embeddings, p=1, dim=1)
    
    # for embedding in KG_embeddings:
    #     print(torch.sum(embedding))
    # sys.exit()
    
    # sys.exit()
    # pdb.set_trace()

    print(len(KG_vocab))
        
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if dataset_name == 'places365':
        dataset = PLACES_365(base_dir, subset_path)
    elif dataset_name == 'ade20k':
        dataset = ADE_20k(base_dir, subset_path)
    else:
        dataset = EPIC_Kitchens(base_dir, subset_path)
    

    dataset_size = len(dataset)
    train_size = int(0.7 * dataset_size)
    test_size = dataset_size - train_size

    train_set, test_set = random_split(dataset, [train_size, test_size])

    
    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8, collate_fn=custom_collate)
    test_dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=8, collate_fn=custom_collate)
    
    # model = build_gsnn(configs,KG_vocab, KG_nodes)
    # model.train()
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)


    model_mgsnn = build_mgsnn(configs,KG_vocab, KG_nodes)
    model_mgsnn.train()
    optimizer = torch.optim.Adam(model_mgsnn.parameters(), lr=lr)
    
    mgnn_loss = MGNNLoss(alpha=alpha)
    MSE_loss = nn.MSELoss()

    KG_embeddings = KG_embeddings.requires_grad_(True)
    KG_adjacency_matrix = KG_adjacency_matrix.requires_grad_(True)


    # For dogsled image
    # img_path = '/data/aryan/Seekg/MGNN/results/image.png'

    # bbox = get_bboxes_labels(img_path)
    # # print(bbox)
    # bbox_formated = []
    # for obj in bbox:
    #     for i, obj in enumerate(bbox[obj]):
    #         print(obj['label'])
    #         if obj['label'] == 'chair':
    #             bbox_formated.append({'object': 'sled'+'_'+str(i), 'object_bbox': obj['bbox']})  
    #         else:
    #             bbox_formated.append({'object': obj['label']+'_'+str(i), 'object_bbox': obj['bbox']})  


    # For garage
    # img_path = '/data/aryan/Seekg/MGNN/results/test_images/garage.jpg'

    # bbox = get_bboxes_labels(img_path)
    # # print(bbox)
    # bbox_formated = []
    # for obj in bbox:
    #     for i, obj in enumerate(bbox[obj]):
    #         print(obj['label'])
    #         # if obj['label'] == 'chair':
    #         #     bbox_formated.append({'object': 'sled'+'_'+str(i), 'object_bbox': obj['bbox']})  
    #         # else:
    #         bbox_formated.append({'object': obj['label']+'_'+str(i), 'object_bbox': obj['bbox']})  


    # #For gym
    # img_path = '/data/aryan/Seekg/MGNN/results/test_images/gym.jpeg'

    # bbox = get_bboxes_labels(img_path)
    # # print(bbox)
    # bbox_formated = []
    # for obj in bbox:
    #     for i, obj in enumerate(bbox[obj]):
    #         print(obj['label'])
    #         # if obj['label'] == 'chair':
    #         #     bbox_formated.append({'object': 'sled'+'_'+str(i), 'object_bbox': obj['bbox']})  
    #         # else:
    #         bbox_formated.append({'object': obj['label']+'_'+str(i), 'object_bbox': obj['bbox']})  


    #For harbor
    # img_path = '/data/aryan/Seekg/MGNN/results/test_images/harbour.jpg'

    # bbox = get_bboxes_labels(img_path)
    # # print(bbox)
    # bbox_formated = []
    # for obj in bbox:
    #     for i, obj in enumerate(bbox[obj]):
    #         print(obj['label'])
    #         # if obj['label'] == 'chair':
    #         #     bbox_formated.append({'object': 'sled'+'_'+str(i), 'object_bbox': obj['bbox']})  
    #         # else:
    #         bbox_formated.append({'object': obj['label']+'_'+str(i), 'object_bbox': obj['bbox']})  

    #For empty room
    img_path = '/data/aryan/Seekg/MGNN/results/test_images_gpt_failure/empty_bedroom.jpg'
    bbox = get_bboxes_labels(img_path)
    # print(bbox)
    bbox_formated = []
    for obj in bbox:
        for i, obj in enumerate(bbox[obj]):
            print(obj['label'])
            # if obj['label'] == 'chair':
            #     bbox_formated.append({'object': 'sled'+'_'+str(i), 'object_bbox': obj['bbox']})  
            # else:
            bbox_formated.append({'object': obj['label']+'_'+str(i), 'object_bbox': obj['bbox']})  
 

    SG_nodes, SG_Adj = generate_SG_from_bboxs(bbox_formated, 10000) 

    ##For garage
    # SG_nodes[3] = 'table_0'
    # SG_nodes[5] = 'sofa_0'
    # SG_nodes[0] = 'wall_0'
    # SG_nodes[4] = 'floor_0'

    # #For gym
    # SG_nodes[3] = 'television_0'

    # For empty Bedroom
    SG_nodes[0] = 'door_0'


    print(SG_nodes)
    # SG_Adj[1,0] =0
    # SG_Adj[0,1] =0
    # visualize_graph(SG_nodes, SG_Adj)

    obj ='door'
    active_idx = get_KG_active_idx(SG_nodes, SG_Adj, KG_vocab , obj)
    active_SG_idx = get_SG_active_idx(SG_nodes, SG_Adj, KG_vocab , obj)
    SG_Adj = update_SG_adj(SG_nodes, SG_Adj, KG_vocab , obj)
    # print(active_idx)

    for node in active_idx:
        print(KG_vocab[node])
    # print("####")
    
    imp, idx = model_mgsnn(KG_embeddings, KG_adjacency_matrix, SG_Adj,[active_idx, active_SG_idx])

    for i, im in enumerate(imp.tolist()):
        if im < 2  :
            idx[i] = 0
    # print(imp)s
    # onehot = torch.where(idx == torch.tensor(KG_vocab.index(verb)), torch.ones_like(imp), torch.zeros_like(imp))
    # print(imp)
    # print(onehot)

    # print(imp.shape)
    # print(onehot.shape)
    # loss = MSE_loss(imp, onehot.float())
    # print(loss)
    # optimizer.zero_grad()

    # loss.backward(retain_graph=True)   
    # optimizer.step()

    GSNN_output =idx[torch.topk(imp, k=1).indices]
    for node in GSNN_output.detach().int():
        if KG_vocab[node] in ["/living_room",'/bedroom', '/kitchen', '/bathroom']:
        # if KG_vocab[node] in ["harbor",'/lake', '/ocean', '/bathroom']:
            print(KG_vocab[node])
        # print(KG_vocab[node])

    # actions = get_actions(torch.cat((GSNN_output.detach().cpu(), active_idx)), KG_path)
    # print(actions)







    # for epoch in tqdm(range(epochs)):
    #     train_accuracy = []
    #     train_precision = []
    #     train_recall = []

    #     for VISOR_bboxs,objs in tqdm(train_dataloader):
    #         verbs = torch.tensor([KG_vocab.index(obj[1]) for obj in objs])
    #         # print(verbs)
    #         GSNN_outputs = []   
    #         if objs[0][1] == 'cut' and objs[0][0] == 'meat':

    #             for  bbox, obj in zip(VISOR_bboxs, objs):
    #                 verb = obj[1]
    #                 obj = obj[0]
    #                 # print("Verb: ", verb)
    #                 SG_nodes, SG_Adj = generate_SG_from_bboxs(bbox, 200) 
    #                 visualize_graph(SG_nodes, SG_Adj)
    #                 print(SG_nodes)
    #                 # print(SG_Adj)

    #                 active_idx = get_KG_active_idx(SG_nodes, SG_Adj, KG_vocab , obj)
    #                 # print(active_idx)

    #                 # for node in active_idx:
    #                 #     print(KG_vocab[node])
    #                 # print("####")
    #                 imp, idx = model(KG_embeddings, KG_adjacency_matrix, active_idx)
    #                 onehot = torch.where(idx == torch.tensor(KG_vocab.index(verb)), torch.ones_like(imp), torch.zeros_like(imp))
    #                 # print(imp)
    #                 # print(onehot)

    #                 # print(imp.shape)
    #                 # print(onehot.shape)
    #                 loss = MSE_loss(imp, onehot.float())
    #                 # print(loss)
    #                 optimizer.zero_grad()

    #                 loss.backward(retain_graph=True)   
    #                 optimizer.step()

    #                 GSNN_output =idx[torch.topk(imp, k=3).indices]
    #                 actions = get_actions(torch.cat((GSNN_output.detach().cpu(), active_idx)), KG_path)
    #                 print(actions)
    #                 # for node in GSNN_output.detach().int():
    #                 #     print(KG_vocab[node])
    #                 GSNN_outputs.append(GSNN_output) 
    #                 sys.exit()



def main():
    parser = argparse.ArgumentParser(description="Training AirLoc")
    parser.add_argument(
        "-c", "--config_file",
        dest = "config_file",
        type = str, 

        default = "configs/train_gsnn.yaml"
    )
    parser.add_argument(
        "-g", "--gpu",
        dest = "gpu",
        type = int, 
        default = [1]
    )
    args = parser.parse_args()

    config_file = args.config_file
    f = open(config_file, 'r', encoding='utf-8')
    configs = f.read()
    configs = yaml.safe_load(configs)
    configs['num_gpu'] = args.gpu

    train(configs)
    
if __name__ == "__main__":
    main()