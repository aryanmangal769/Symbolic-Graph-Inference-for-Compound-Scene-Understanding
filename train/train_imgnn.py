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

from model.build_model import build_gsnn , build_mgsnn, build_vit , build_imgsnn
from model.mgnn.mgnn_loss import MGNNLoss
from datasets.epic_kitchens import EPIC_Kitchens
from datasets.places_365 import PLACES_365
from datasets.ade20k import ADE_20k
from utils.dataset_utils import custom_collate
from utils.scene_graph_utils import generate_SG_from_bboxs, get_KG_active_idx, visualize_graph, get_SG_active_idx, generate_SG
from utils.vis_utils import visualize_bbox
import numpy as np
import pdb
from sklearn.metrics import f1_score
import seaborn as sns
from sklearn.metrics import confusion_matrix
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import random
import os


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

def train(configs):
    dataset_name = configs['dataset']
    base_dir = configs['base_dir']
    subset_path = configs['subset_path']
    epochs = configs['epochs']
    batch_size = configs['batch_size']
    alpha = configs['alpha']
    lr = configs['lr']

    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),])

   
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
    
    random.seed(40)
    torch.manual_seed(40)

    dataset_size = len(dataset)
    train_size = int(0.7 * dataset_size)
    test_size = dataset_size - train_size

    train_set, test_set = random_split(dataset, [train_size, test_size])

    
    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8, collate_fn=custom_collate)
    test_dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=8, collate_fn=custom_collate)
    
    num_classes = len(KG_nodes['tools'][0])
    configs['vit']['num_classes'] = num_classes
    configs['vit']['head'] = False

    model_mgsnn = build_imgsnn(configs,KG_vocab, KG_nodes)
    model_mgsnn.train()
    optimizer = torch.optim.Adam(model_mgsnn.parameters(), lr=lr)

    model_vit = build_vit(configs)
    model_vit.eval()

    ce_loss = nn.CrossEntropyLoss()    
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
                # print("Verb: ", verb)

                # SG_nodes, SG_Adj = generate_SG_from_bboxs(bbox, 1000) 
                # SG_nodes, SG_embeddings, SG_Adj = generate_SG(bbox, 1000, img, model_vit) 
                SG_nodes, SG_embeddings, SG_Adj = generate_SG(bbox, 400, img, model_vit) 
                # print(SG_nodes)
                # visualize_graph(SG_nodes, SG_Adj)


                active_idx = get_KG_active_idx(SG_nodes, SG_Adj, KG_vocab , obj)
                active_SG_idx = get_SG_active_idx(SG_nodes, SG_Adj, KG_vocab , obj)
                
                # Only used when we want (Save some initial active nodes get the results and merge them)
                active_idx_1 = get_KG_active_idx(SG_nodes, SG_Adj, KG_vocab , SG_nodes[0].split('_')[0])
                active_SG_idx_1 = get_SG_active_idx(SG_nodes, SG_Adj, KG_vocab , SG_nodes[0].split('_')[0])

                # for node in active_idx:
                #     print(KG_vocab[node])
                # print("####")

                img_feat = model_vit(transform(img).unsqueeze(0))

                imp, idx = model_mgsnn(KG_embeddings, KG_adjacency_matrix, SG_Adj,SG_embeddings, [active_idx, active_SG_idx], img_feat)

                # Only used when we want (Save some initial active nodes get the results and merge them)
                imp_1, idx_1 = model_mgsnn(KG_embeddings, KG_adjacency_matrix, SG_Adj,SG_embeddings, [active_idx_1, active_SG_idx_1], img_feat)
                if imp_1.max() > imp.max():
                    imp = imp_1
                    idx = idx_1


                imp = imp.squeeze(0)
                onehot = torch.tensor([float(tool == verb) for tool in KG_nodes['tools'][0]], dtype=torch.float32).to(imp.device)
                
                loss = ce_loss(imp, onehot)
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()

                idx_all = torch.tensor([KG_vocab.index(verb) for verb in KG_nodes['tools'][0]]).to(imp.device)

                # for i, id in enumerate(idx_all):
                #     if id not in idx:
                #         imp[i] = 0

                GSNN_output =idx_all[torch.topk(imp, k=1).indices]

                # for node in idx:
                #     print(KG_vocab[node])
                # print(onehot)

                
                if dataset_name != 'places365': 
                    actions = get_actions(torch.cat((GSNN_output.detach().cpu(), active_idx)), KG_path)
                    # print(actions)
                    # for node in GSNN_output.detach().int():
                    #     print(KG_vocab[node])

                if KG_vocab[GSNN_output[0]] != 'None':
                    predicted_verbs.append(KG_vocab[GSNN_output[0]])
                else:
                    predicted_verbs.append(KG_vocab[GSNN_output[1]])

                actual_verbs.append(verb)

                GSNN_outputs.append(GSNN_output) 

            GSNN_outputs = torch.stack(GSNN_outputs, dim=1)        
            train_accuracy.append(get_accuracy(GSNN_outputs.squeeze(1).float(), verbs.to(GSNN_outputs.device).float()))

        print("Epoch: ", epoch, " Loss: ", loss.item(), "Train Accuracy: ", sum(train_accuracy)/len(train_accuracy))
        labels = np.unique(actual_verbs)
        print(labels )
        print("F1 score: ", f1_score(actual_verbs, predicted_verbs,labels = labels, average=None))

        test_accuracy = []

        predicted_verbs = []
        actual_verbs = []
        active_idxs = []

        verb_counts = {}

        for VISOR_bboxs,objs, imgs in tqdm(test_dataloader):
            verbs = torch.tensor([KG_vocab.index(obj[1]) for obj in objs])

            # current_verb = verbs[0].item()
            # if current_verb in verb_counts:
            #     verb_counts[current_verb] += 1
            # else:
            #     verb_counts[current_verb] = 1

            # if verb_counts[current_verb] > 5:
            #     continue

            # Get the image
            # img = imgs[0]

            # Save the image to the corresponding verb folder
            # verb_folder = os.path.join('results/gpt_vision', KG_vocab[current_verb].replace('/', ''))
            # os.makedirs(verb_folder, exist_ok=True)
            # img_path = os.path.join(verb_folder, f"image_{verb_counts[current_verb]}.png")
            # img_pil = transforms.ToPILImage()(img)
            # img_pil.save(img_path)



            # print(verbs)
            GSNN_outputs = []  

            for  bbox, obj, img in zip(VISOR_bboxs, objs, imgs):
                verb = obj[1]
                obj = obj[0]
                
                # print("Verb: ", verb)

                # SG_nodes, SG_Adj = generate_SG_from_bboxs(bbox, 1000) 
                # print(SG_nodes)
                # visualize_graph(SG_nodes, SG_Adj)
                # active_idxs.append(SG_nodes)
                # SG_nodes, SG_embeddings, SG_Adj = generate_SG(bbox, 1000, img, model_vit)
                SG_nodes, SG_embeddings, SG_Adj = generate_SG(bbox, 400, img, model_vit)

                active_idx = get_KG_active_idx(SG_nodes, SG_Adj, KG_vocab , obj)
                active_SG_idx = get_SG_active_idx(SG_nodes, SG_Adj, KG_vocab , obj)

                # Only used when we want (Save some initial active nodes get the results and merge them)
                active_idx_1 = get_KG_active_idx(SG_nodes, SG_Adj, KG_vocab , SG_nodes[0])
                active_SG_idx_1 = get_SG_active_idx(SG_nodes, SG_Adj, KG_vocab , SG_nodes[0])

                act_arr = []
                for node in active_idx:
                    act_arr.append(KG_vocab[node])
                    # print(KG_vocab[node])
                active_idxs.append(act_arr)

                img_feat = model_vit(transform(img).unsqueeze(0))

                imp, idx = model_mgsnn(KG_embeddings, KG_adjacency_matrix, SG_Adj, SG_embeddings, [active_idx, active_SG_idx], img_feat)

                # Only used when we want (Save some initial active nodes get the results and merge them)
                imp_1, idx_1 = model_mgsnn(KG_embeddings, KG_adjacency_matrix, SG_Adj,SG_embeddings, [active_idx_1, active_SG_idx_1], img_feat)
                if imp_1.max() > imp.max():
                    imp = imp_1
                    idx = idx_1

                imp = imp.squeeze(0)    
                onehot = torch.tensor([float(tool == verb) for tool in KG_nodes['tools'][0]], dtype=torch.float32).to(imp.device)
                
                loss = ce_loss(imp, onehot)

                idx_all = torch.tensor([KG_vocab.index(verb) for verb in KG_nodes['tools'][0]]).to(imp.device)
                
                # for i, id in enumerate(idx_all):
                #     if id not in idx:
                #         imp[i] = 0
                
                GSNN_output =idx_all[torch.topk(imp, k=1).indices]
            


                if KG_vocab[GSNN_output[0]] != 'None':
                    predicted_verbs.append(KG_vocab[GSNN_output[0]])
                else:
                    predicted_verbs.append(KG_vocab[GSNN_output[1]])

                # predicted_verbs.append([[KG_vocab[GSNN_output[i]] for i in range(2)], torch.topk(imp, k=6).values])
                actual_verbs.append(verb)


                # actions = get_actions(torch.cat((GSNN_output.detach().cpu(), active_idx)), KG_path)
                # # print(actions)
                # for node in GSNN_output.detach().int():
                #     print(KG_vocab[node])


                
                GSNN_outputs.append(GSNN_output) 

                # sys.exit()
                # print("####")
            
            GSNN_outputs = torch.stack(GSNN_outputs, dim=1)
            # print("Accuracy", get_accuracy(GSNN_outputs.squeeze(1).float(), verbs.to(GSNN_outputs.device).float()))

            acc = get_accuracy(GSNN_outputs.squeeze(1).float(), verbs.to(GSNN_outputs.device).float())
            # if acc < 1 and epoch == 2:
            #     img = imgs[0]
            #     verb_folder = os.path.join('results/mgnn_wrong', KG_vocab[verbs[0].item()].replace('/', ''))
            #     os.makedirs(verb_folder, exist_ok=True)
            #     img_path = os.path.join(verb_folder, f"image_{acc}.png")
            #     # img_pil = transforms.ToPILImage()(img)
            #     img.save(img_path)

            test_accuracy.append(get_accuracy(GSNN_outputs.squeeze(1).float(), verbs.to(GSNN_outputs.device).float()))

        
        with open("results/chat_gpt_test.pkl", 'wb') as f:
            pickle.dump([active_idxs,  actual_verbs], f)        
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
        default = [2]
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