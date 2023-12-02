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
            embeddings.append(torch.zeros(glove.dim))
    return torch.stack(embeddings)

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

    with open("images_per_class.pkl", "rb") as file:
        images_per_class, top_objects_per_class = pickle.load(file)

    for category in top_objects_per_class.keys():
        objects = top_objects_per_class[category]

        objects = Counter(objects).most_common()    
        for i, obj in enumerate(objects):
            if i < 10:
                tools.append([category, obj[0]])
            relations.append(['None', obj[0]])
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
        'objects': [list(set([item for sublist in relations for item in sublist]))],
        'affordances': [list(set([item[1] for item in affordances]))],
        'tools': [list(set([item[1] for item in tools]))],
    }

    with open('/data/aryan/Seekg/Datasets/places365/places365places_KG.pkl', 'wb') as f:
        pickle.dump((embeddings, adjacency_matrix, vocab,nodes), f)
    



if (__name__ == '__main__'):
    makeGraph()