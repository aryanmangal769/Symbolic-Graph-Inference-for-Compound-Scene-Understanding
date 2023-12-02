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
from model.mgnn.mgnn_loss import MGNNLoss
from datasets.epic_kitchens import EPIC_Kitchens
from utils.dataset_utils import custom_collate
from utils.scene_graph_utils import generate_SG_from_bboxs, get_KG_active_idx, visualize_graph
from utils.vis_utils import visualize_bbox
import numpy as np


def get_actions(idx, KG_path ):
    actions = []
    with open(KG_path, 'rb') as f:
        KG_embeddings, KG_adjacency_matrix, KG_vocab, KG_nodes = pickle.load(f)
    # print(len(KG_vocab))
    verbs = []
    for i in idx:
        # print(KG_vocab[i])
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

def train(configs):
    base_dir = configs['base_dir']
    subset_path = configs['subset_path']
    epochs = configs['epochs']
    batch_size = configs['batch_size']
    alpha = configs['alpha']
    lr = configs['lr']

   
    KG_path = configs['KG_path']    
    with open(KG_path, 'rb') as f:
        KG_embeddings, KG_adjacency_matrix, KG_vocab, KG_nodes = pickle.load(f)
    # print(len(KG_vocab))
        
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    dataset = EPIC_Kitchens(base_dir, subset_path)
    dataset_size = len(dataset)
    train_size = int(0.7 * dataset_size)
    test_size = dataset_size - train_size

    train_set, test_set = random_split(dataset, [train_size, test_size])
    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=1, collate_fn=custom_collate)
    test_dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=1, collate_fn=custom_collate)
    
    model = build_gsnn(configs)
    model.train()
    
    mgnn_loss = MGNNLoss(alpha=alpha)
    MSE_loss = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    KG_embeddings = KG_embeddings.requires_grad_(True)
    KG_adjacency_matrix = KG_adjacency_matrix.requires_grad_(True)

    for epoch in tqdm(range(epochs)):
        train_accuracy = []
        train_precision = []
        train_recall = []

        for VISOR_bboxs,objs in tqdm(train_dataloader):
            verbs = torch.tensor([KG_vocab.index(obj[1]) for obj in objs])
            # print(verbs)
            GSNN_outputs = []   
            for  bbox, obj in zip(VISOR_bboxs, objs):
                verb = obj[1]
                obj = obj[0]
                # print("Verb: ", verb)
                SG_nodes, SG_Adj = generate_SG_from_bboxs(bbox, 200) 
                # visualize_graph(SG_nodes, SG_Adj)
                # print(SG_nodes)
                # print(SG_Adj)

                # active_idx = get_KG_active_idx(SG_nodes, SG_Adj, KG_vocab , obj)
                # print(active_idx)

                # for node in active_idx:
                #     print(KG_vocab[node])
                # print("####")
                active_idx = []
                for i in SG_nodes:
                    if i in KG_vocab:
                        active_idx.append(KG_vocab.index(i))
                
                active_idx = torch.tensor(list(set(active_idx)))
                # print(active_idx)
                imp, idx = model(KG_embeddings, KG_adjacency_matrix, active_idx)
                onehot = torch.where(idx == torch.tensor(KG_vocab.index(verb)), torch.ones_like(imp), torch.zeros_like(imp))
                # print(imp)
                # print(onehot)

                # print(imp.shape)
                # print(onehot.shape)
                loss = MSE_loss(imp, onehot.float())
                # print(loss)
                optimizer.zero_grad()

                loss.backward(retain_graph=True)   
                optimizer.step()

                try : 
                    GSNN_output =idx[torch.topk(imp, k=3).indices]
                except:
                    GSNN_output =idx[[0,0,0]]
                # GSNN_output =idx[torch.topk(imp, k=2).indices]
                actions = get_actions(torch.cat((GSNN_output.detach().cpu(), active_idx)), KG_path)
                # print(actions)
                # for node in GSNN_output.detach().int():
                #     print(KG_vocab[node])
                GSNN_outputs.append(GSNN_output) 
            # break
            
            GSNN_outputs = torch.stack(GSNN_outputs, dim=1)

            # loss = MSE_loss(GSNN_outputs.squeeze(1).float(), verbs.to(GSNN_outputs.device).float())
            # print(GSNN_outputs.reshape((GSNN_outputs.shape[1],GSNN_outputs.shape[0])).float().requires_grad)
            # loss = mgnn_loss(GSNN_outputs.reshape((GSNN_outputs.shape[1],GSNN_outputs.shape[0])).float(), verbs.to(GSNN_output.device).float())
            # print(loss)

            # optimizer.zero_grad()

            # loss.backward(retain_graph=True)   
            # optimizer.step()

            precision, recall = get_precision_recall(GSNN_outputs, verbs)
            train_precision.append(precision)
            train_recall.append(recall)
        
            train_accuracy.append(get_accuracy(GSNN_outputs.squeeze(1).float(), verbs.to(GSNN_outputs.device).float()))
            # print(accuracy[-1])
        average_train_precision = sum(train_precision) / len(train_precision)
        average_train_recall = sum(train_recall) / len(train_recall)
        
        print(f"Epoch: {epoch}, Loss: {loss.item()}, Train Precision: {average_train_precision}, Train Recall: {average_train_recall}")
        print("Epoch: ", epoch, " Loss: ", loss.item(), "Train Accuracy: ", sum(train_accuracy)/len(train_accuracy))

        test_accuracy = []
        test_precision = []
        test_recall = []

        for VISOR_bboxs,objs in tqdm(test_dataloader):
            verbs = torch.tensor([KG_vocab.index(obj[1]) for obj in objs])
            # print(verbs)
            GSNN_outputs = []   
            # if objs[0][1] == 'shake' and objs[0][0] == 'cup':

            for  bbox, obj in zip(VISOR_bboxs, objs):
                verb = obj[1]
                obj = obj[0]
                # print("Verb: ", verb)
                SG_nodes, SG_Adj = generate_SG_from_bboxs(bbox, 200) 
                # visualize_graph(SG_nodes, SG_Adj)
                # print(SG_nodes)
                # print(SG_Adj)

                active_idx = get_KG_active_idx(SG_nodes, SG_Adj, KG_vocab , obj)
                # print(active_idx)

                # for node in active_idx:
                #     print(KG_vocab[node])
                # print("####")
                imp, idx = model(KG_embeddings, KG_adjacency_matrix, active_idx)
                onehot = torch.where(idx == torch.tensor(KG_vocab.index(verb)), torch.ones_like(imp), torch.zeros_like(imp))
                # print(imp)
                # print(onehot)

                # print(imp.shape)
                # print(onehot.shape)
                loss = MSE_loss(imp, onehot.float())
                # print(loss)

                GSNN_output =idx[torch.topk(imp, k=3).indices]
                actions = get_actions(torch.cat((GSNN_output.detach().cpu(), active_idx)), KG_path)
                # print(actions)
                # for node in GSNN_output.detach().int():
                #     print(KG_vocab[node])
                GSNN_outputs.append(GSNN_output) 
            # break
            
            GSNN_outputs = torch.stack(GSNN_outputs, dim=1)

            # loss = MSE_loss(GSNN_outputs.squeeze(1).float(), verbs.to(GSNN_outputs.device).float())
            # print(GSNN_outputs.reshape((GSNN_outputs.shape[1],GSNN_outputs.shape[0])).float().requires_grad)
            # loss = mgnn_loss(GSNN_outputs.reshape((GSNN_outputs.shape[1],GSNN_outputs.shape[0])).float(), verbs.to(GSNN_output.device).float())
            # print(loss)

            # optimizer.zero_grad()

            # loss.backward(retain_graph=True)   
            # optimizer.step()

            precision, recall = get_precision_recall(GSNN_outputs, verbs)
            test_precision.append(precision)
            test_recall.append(recall)
        
            test_accuracy.append(get_accuracy(GSNN_outputs.squeeze(1).float(), verbs.to(GSNN_outputs.device).float()))
            # print(accuracy[-1])
        average_test_precision = sum(test_precision) / len(test_precision)
        average_test_recall = sum(test_recall) / len(test_recall)
        
        print(f"Epoch: {epoch}, Loss: {loss.item()}, test Precision: {average_test_precision}, test Recall: {average_test_recall}")
        print("Epoch: ", epoch, " Loss: ", loss.item(), "test Accuracy: ", sum(test_accuracy)/len(test_accuracy))
        
            

            # verb = objs[0][1]
            # bbox = VISOR_bboxs[0]
            # obj = objs[0][0]
            # print("Verb: ", objs[0][1])

            # # visualize_bbox(img, bbox, 'img_with_bbox.jpg')   
            
            # SG_nodes, SG_Adj = generate_SG_from_bboxs(bbox, 200) 
            # print(SG_nodes)
            # print(SG_Adj)

            # active_idx = get_KG_active_idx(SG_nodes, SG_Adj, KG_vocab , obj)
            # print(active_idx)
            # for node in active_idx:
            #     print(KG_vocab[node])
            # print("####")
            # GSNN_output = model(KG_embeddings, KG_adjacency_matrix, active_idx)
            # for node in GSNN_output:
            #     print(KG_vocab[node])
            # # print(model(KG_embeddings, KG_adjacency_matrix, active_idx))
            # break



    # VISOR_bboxs, objs = dataloader.__iter__().__next__()

    # # img = cv2.imread(img_paths[0])
    # bbox = VISOR_bboxs[0]
    # obj = objs[0][0]
    # print("Verb: ", objs[0][1])

    # # visualize_bbox(img, bbox, 'img_with_bbox.jpg')   
    
    # SG_nodes, SG_Adj = generate_SG_from_bboxs(bbox, 200) 
    # print(SG_nodes)
    # print(SG_Adj)
    
    # active_idx = get_SG_active_idx(SG_nodes, SG_Adj, KG_vocab , obj)
    # print(active_idx)
    # print(model(KG_embeddings, KG_adjacency_matrix, active_idx))





    # A = [[0, 1, 0, 0, 0, 0],
    #     [1, 0, 1, 0, 0, 0],
    #     [0, 1, 0, 1, 1, 0],
    #     [0, 0, 1, 0, 1, 0],
    #     [0, 0, 1, 1, 0, 1],
    #     [0, 0, 0, 0, 1, 0]]
    # A = torch.tensor(A, dtype=torch.float32)

    # x = torch.randn(6, configs['gcn']['n_feat'])

    # active_idx = [2,4]
    # active_idx = torch.tensor(active_idx, dtype=torch.int64)

    # print(model(x, A, active_idx))


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