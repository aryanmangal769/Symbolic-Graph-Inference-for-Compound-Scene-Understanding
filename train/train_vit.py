import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.data import random_split
import torch.nn.functional as F
import sys
sys.path.append('.')
import argparse 
import yaml
import pickle
from tqdm import tqdm

from model.build_model import build_gsnn , build_mgsnn, build_vit
from model.mgnn.mgnn_loss import MGNNLoss
from datasets.epic_kitchens import EPIC_Kitchens
from datasets.places_365 import PLACES_365
from datasets.ade20k import ADE_20k
from utils.dataset_utils import custom_collate
from utils.scene_graph_utils import generate_SG_from_bboxs, get_KG_active_idx, visualize_graph, get_SG_active_idx
from utils.vis_utils import visualize_bbox
import numpy as np
import pdb
from sklearn.metrics import f1_score
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


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

    print(len(KG_vocab))
        
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if dataset_name == 'places365':
        dataset = PLACES_365(base_dir, subset_path)
    elif dataset_name == 'ade20k':
        dataset = ADE_20k(base_dir, configs['data_dir'])
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


    # model_mgsnn = build_mgsnn(configs,KG_vocab, KG_nodes)
    # model_mgsnn.train()
    # optimizer = torch.optim.Adam(model_mgsnn.parameters(), lr=lr)

    num_classes = len(KG_nodes['tools'][0])
    configs['vit']['num_classes'] = num_classes
    configs['vit']['head'] = True
    model_vit = build_vit(configs)
    model_vit.train()
    for name, param in model_vit.named_parameters():
        print(name, param.shape)
        if 'head' not in name:  # Identify the last head layer
            param.requires_grad = False

    ce_loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model_vit.parameters(), lr=1e-4)
    
    mgnn_loss = MGNNLoss(alpha=alpha)
    MSE_loss = nn.MSELoss()

    KG_embeddings = KG_embeddings.requires_grad_(True)
    KG_adjacency_matrix = KG_adjacency_matrix.requires_grad_(True)

    for epoch in tqdm(range(epochs)):
        train_accuracy = []

        predicted_verbs = []
        actual_verbs = []

        for VISOR_bboxs,objs,imgs in tqdm(train_dataloader):
            verbs = torch.tensor([KG_vocab.index(obj[1]) for obj in objs])

            GSNN_outputs = []   
            for  bbox, obj , img in zip(VISOR_bboxs, objs , imgs):
                verb = obj[1]
                obj = obj[0]
                # # print("Verb: ", verb)

                # SG_nodes, SG_Adj = generate_SG_from_bboxs(bbox, 1000) 
                # # print(SG_nodes)
                # # visualize_graph(SG_nodes, SG_Adj)


                # active_idx = get_KG_active_idx(SG_nodes, SG_Adj, KG_vocab , obj)
                # active_SG_idx = get_SG_active_idx(SG_nodes, SG_Adj, KG_vocab , obj)
                # print(active_idx)

                # for node in active_idx:
                #     print(KG_vocab[node])
                # print("####")
                # imp, idx = model_mgsnn(KG_embeddings, KG_adjacency_matrix, SG_Adj,[active_idx, active_SG_idx])
                # imp, idx = model(KG_embeddings, KG_adjacency_matrix, active_idx)
                # print(imp)
                # onehot = torch.where(idx == torch.tensor(KG_vocab.index(verb)), torch.ones_like(imp), torch.zeros_like(imp))
                # for node in idx:
                #     print(KG_vocab[node])
                # print(onehot)

                # print(imp)
                # print(onehot)
                # loss = MSE_loss(imp, onehot.float())
                # optimizer.zero_grad()
                # loss.backward(retain_graph=True)   
                # optimizer.step()

                # if imp.shape[0] < 6:
                #     imp = torch.cat((imp, torch.zeros(6, dtype=torch.float32).to(imp.device)))
                #     idx = torch.cat((idx, torch.zeros(6, dtype=torch.int64).to(idx.device)))
                # GSNN_output =idx[torch.topk(imp, k=1).indices]
                
                # if dataset_name != 'places365': 
                #     actions = get_actions(torch.cat((GSNN_output.detach().cpu(), active_idx)), KG_path)
                #     # print(actions)
                #     # for node in GSNN_output.detach().int():
                #     #     print(KG_vocab[node])

                # if KG_vocab[GSNN_output[0]] != 'None':
                #     predicted_verbs.append(KG_vocab[GSNN_output[0]])
                # else:
                #     predicted_verbs.append(KG_vocab[GSNN_output[1]])

                vit_output = model_vit(img.unsqueeze(0))
                vit_output = vit_output.squeeze(0)
                # print(KG_nodes['tools'][0])
                # pdb.set_trace()
                onehot = torch.tensor([float(tool == verb) for tool in KG_nodes['tools'][0]], dtype=torch.float32).to(vit_output.device)
                # onehot = torch.where((torch.tensor(verb) == KG_nodes['tools'][0] ), torch.ones_like(vit_output), torch.zeros_like(vit_output))
                # onehot = torch.where(KG_nodes['affordances'][0] == verb)
                loss = ce_loss(vit_output, onehot)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                idx_vit = torch.tensor([KG_vocab.index(verb) for verb in KG_nodes['tools'][0]]).to(vit_output.device)
                GSNN_output =idx_vit[torch.topk(vit_output, k=1).indices]


                actual_verbs.append(verb)

                GSNN_outputs.append(GSNN_output) 

                

            GSNN_outputs = torch.stack(GSNN_outputs, dim=1)        
            # print(get_accuracy(GSNN_outputs.squeeze(1).float(), verbs.to(GSNN_outputs.device).float()))
            train_accuracy.append(get_accuracy(GSNN_outputs.squeeze(1).float(), verbs.to(GSNN_outputs.device).float()))

        print("Epoch: ", epoch, " Loss: ", loss.item(), "Train Accuracy: ", sum(train_accuracy)/len(train_accuracy))
        labels = np.unique(actual_verbs)
        print(labels )
        # print("F1 score: ", f1_score(actual_verbs, predicted_verbs,labels = labels, average=None))

        torch.save(model_vit.state_dict(), "results/vit.pth")

        test_accuracy = []

        predicted_verbs = []
        actual_verbs = []
        active_idxs = []

        for VISOR_bboxs,objs,imgs in tqdm(test_dataloader):
            verbs = torch.tensor([KG_vocab.index(obj[1]) for obj in objs])
            # print(verbs)
            GSNN_outputs = []  

            for  bbox, obj,img in zip(VISOR_bboxs, objs,imgs):
                verb = obj[1]
                obj = obj[0]
                
                # # print("Verb: ", verb)

                # SG_nodes, SG_Adj = generate_SG_from_bboxs(bbox, 1000) 
                # # print(SG_nodes)
                # # visualize_graph(SG_nodes, SG_Adj)
                # # active_idxs.append(SG_nodes)

                # active_idx = get_KG_active_idx(SG_nodes, SG_Adj, KG_vocab , obj)
                # active_SG_idx = get_SG_active_idx(SG_nodes, SG_Adj, KG_vocab , obj)

                # # print(active_idx)

                # act_arr = []
                # for node in active_idx:
                #     act_arr.append(KG_vocab[node])
                #     # print(KG_vocab[node])
                # active_idxs.append(act_arr)

                # imp, idx = model_mgsnn(KG_embeddings, KG_adjacency_matrix, SG_Adj,[active_idx, active_SG_idx])
                # # imp, idx = model(KG_embeddings, KG_adjacency_matrix, active_idx)

                # for i, node in enumerate(idx):
                #     if KG_vocab[node] in KG_nodes['objects'][0]:
                #        imp[i] = 0 
                
                # # print(imp)

                # onehot = torch.where(idx == torch.tensor(KG_vocab.index(verb)), torch.ones_like(imp), torch.zeros_like(imp))


                # loss = MSE_loss(imp, onehot.float())

                # if imp.shape[0] < 6:
                #     imp = torch.cat((imp, torch.zeros(6, dtype=torch.float32).to(imp.device)))
                #     idx = torch.cat((idx, torch.zeros(6, dtype=torch.int64).to(idx.device)))

                # GSNN_output =idx[torch.topk(imp, k=5).indices]

                # if KG_vocab[GSNN_output[0]] != 'None':
                #     predicted_verbs.append(KG_vocab[GSNN_output[0]])
                # else:
                #     predicted_verbs.append(KG_vocab[GSNN_output[1]])

                # # predicted_verbs.append([[KG_vocab[GSNN_output[i]] for i in range(2)], torch.topk(imp, k=6).values])
                # actual_verbs.append(verb)


                # actions = get_actions(torch.cat((GSNN_output.detach().cpu(), active_idx)), KG_path)
                # # # print(actions)
                # # for node in GSNN_output.detach().int():
                # #     print(KG_vocab[node])


                vit_output = model_vit(img.unsqueeze(0))
                vit_output = vit_output.squeeze(0)
                # print(KG_nodes['tools'][0])
                # pdb.set_trace()
                onehot = torch.tensor([float(tool == verb) for tool in KG_nodes['tools'][0]], dtype=torch.float32).to(vit_output.device)
                # onehot = torch.where((torch.tensor(verb) == KG_nodes['tools'][0] ), torch.ones_like(vit_output), torch.zeros_like(vit_output))
                # onehot = torch.where(KG_nodes['affordances'][0] == verb)
                loss = ce_loss(vit_output, onehot)
                idx_vit = torch.tensor([KG_vocab.index(verb) for verb in KG_nodes['tools'][0]]).to(vit_output.device)
                GSNN_output =idx_vit[torch.topk(vit_output, k=1).indices]
                GSNN_outputs.append(GSNN_output) 

                # sys.exit()
                # print("####")
            
            GSNN_outputs = torch.stack(GSNN_outputs, dim=1)
            test_accuracy.append(get_accuracy(GSNN_outputs.squeeze(1).float(), verbs.to(GSNN_outputs.device).float()))

        
        # with open("results/chat_gpt_test.pkl", 'wb') as f:
        #     pickle.dump([active_idxs,  actual_verbs], f)        
        print("Epoch: ", epoch, " Loss: ", loss.item(), "test Accuracy: ", sum(test_accuracy)/len(test_accuracy))
        labels = np.unique(actual_verbs)
        print(labels )
        # print("F1 score: ", f1_score(actual_verbs, predicted_verbs,labels = labels, average=None))
        # cm = confusion_matrix(actual_verbs, predicted_verbs, labels=labels)
        # plt.figure(figsize=(10,10))
        # sns.heatmap(cm, annot=False, linewidths=.5, square=True, cmap='Blues_r',
        #             xticklabels=labels,
        #             yticklabels=labels)
        # plt.ylabel('Actual label')
        # plt.xlabel('Predicted label')
        # plt.title('Confusion matrix')
        # plt.savefig("results/confusion_matrix.png")


        # sys.exit()


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