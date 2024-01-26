import torch
import numpy as np
import pickle
from torchtext.vocab import GloVe
from typing import List, Tuple
import pdb
from collections import defaultdict, Counter


def get_embeddings(vocab: List[str], glove: GloVe) -> Tuple[torch.Tensor, List[str]]:
    embeddings = []
    for word in vocab:
        if word in glove.stoi:
            embeddings.append(torch.tensor(glove.vectors[glove.stoi[word]]))
        else:
            embeddings.append(torch.rand(glove.dim)
)
    return torch.stack(embeddings)


# def get_embeddings(vocab: List[str], glove: GloVe) -> Tuple[torch.Tensor, List[str]]:
#     embeddings = []
#     embeddings = torch.rand(len(vocab), glove.dim)
#     return torch.abs(embeddings)

def get_adjcency_matrix(relations: List[List[str]], vocab: List[str]) -> torch.Tensor:
    adjacency_matrix = torch.zeros((len(vocab), len(vocab)))
    for relation in relations:
        adjacency_matrix[vocab.index(relation[0])][vocab.index(relation[1])] = 1
        adjacency_matrix[vocab.index(relation[1])][vocab.index(relation[0])] = 1
    return adjacency_matrix


def getKitchenRelation_automatic(dataset_path):

    relations = []
    affordances = []
    tools = []

    with open("/data/aryan/Seekg/MGNN/datasets/ade20k/scene_objects_counter.pkl", "rb") as file: 
        top_objects_per_class = pickle.load(file)

    for category in top_objects_per_class.keys():
        objects = top_objects_per_class[category].most_common(15)   
        print(category)
        print(objects)
        # print(objects) 
        for i, obj in enumerate(objects):
            if i < 11:
                tools.append([category, obj[0]])
        
        for i , obj in enumerate(objects):
            if i < 11:
                for j in range(i+1, len(objects)):
                    if j < 11:
                        relations.append([obj[0], objects[j][0]])
            
            # relations.append(['None', obj[0]])
        # tools.append(['dogsled_multiple','dog'])
        # tools.append(['dogsled_multiple','sled'])
    return relations, affordances, tools


def makeGraph():
    dataset_pkl_path = ''
    # relations, affordances, tools = getKitchenRelation()
    relations, affordances, tools = getKitchenRelation_automatic(dataset_pkl_path)
    vocab = list(set([item for sublist in relations + affordances + tools for item in sublist]))
    glove = GloVe(name='6B', dim=100)
    embeddings = get_embeddings(vocab, glove)
    adjacency_matrix = get_adjcency_matrix(relations + affordances + tools, vocab )
    nodes = {
        'objects': [list(set([item for sublist in relations for item in sublist]+[sublist[0] for sublist in affordances]+[sublist[1] for sublist in tools]))],
        'affordances': [list(set([item[1] for item in affordances]))],
        'tools': [list(set([item[0] for item in tools]))],
    }

    with open('/data/aryan/Seekg/Datasets/ade20k/ade_KG.pkl', 'wb') as f:
        pickle.dump((embeddings, adjacency_matrix, vocab,nodes), f)
    



if (__name__ == '__main__'):
    makeGraph()