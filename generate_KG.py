import torch
import numpy as np
import pickle
from torchtext.vocab import GloVe
from typing import List, Tuple

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

def getKitchenRelation():
    relations = [
        ['meat', 'frying pan'],
        ['meat', 'spoon'],
        ['meat','knife'],
        ['meat', 'chopping board'],
        ['pan', 'left hand'],
        ['pan', 'right hand'],
        ['pan', 'spatula'],
        ['pan', 'fork'],
        ['pan', 'spoon'],
        ['pan', 'plate'],
        ['pan', 'tap'],
        ['pan', 'sink'],
        ['pan', 'water'],
        ['pan', 'hob/cooktop/stovetop'],
        ['pan', 'spoon'],
        ['rice', 'spoon'],
        ['rice', 'pan'],
        ['rice', 'spatula'],
        ['cup', 'spoon'],
        ['cup', 'tea'],
        ['cup', 'tap'],
        ['cup', 'sink'],
        ['cup', 'water'],
        ['cup', 'plate'],
        ['cup', 'sugar'],
        ['cup', 'milk container'],
        ['cup', 'milk'],
        ['cup', 'tea bag'],  
    ]
    affordances = [
        ['meat', 'cut'],
        ['meat', 'mix'],
        ['cup', 'shake'],
        ['cup', 'wash'],
        ['rice', 'pour'],
        ['rice', 'mix'],
        ['pan', 'insert'],
        ['pan', 'wash'],
        
    ]
    tools = [
        ['wash', 'tap'],
        ['wash', 'sink'],
        ['wash', 'water'],
        ['cut', 'knife'],
        ['cut', 'chopping board'],
        ['mix', 'spoon'],
        ['mix', 'frying pan'],
        ['mix', 'spatula'],
        ['insert', 'meat'],
        ['insert', 'egg'],
        ['insert', 'vegetable'],
        ['insert', 'rice'],
        ['insert', 'hob/cooktop/stovetop'],
        ['insert', 'hob'],
        ['insert', 'cooktop'],
        ['insert', 'stovetop'],
        ['shake', 'spoon'],
        ['pour', 'rice container'],
        ['pour', 'cup'],


        
    ]
    return relations, affordances, tools

def makeGraph():
    relations, affordances, tools = getKitchenRelation()
    vocab = list(set([item for sublist in relations + affordances + tools for item in sublist]))
    glove = GloVe(name='6B', dim=100)
    embeddings = get_embeddings(vocab, glove)
    adjacency_matrix = get_adjcency_matrix(relations + affordances + tools, vocab )
    nodes = {
        'objects': [list(set([item for sublist in relations for item in sublist]))],
        'affordances': [list(set([item[1] for item in affordances]))],
        'tools': [list(set([item[1] for item in tools]))],

    }

    with open('/data/aryan/Seekg/Datasets/epic_kitchens_affordances/data/epic_kitchens_KG.pkl', 'wb') as f:
        pickle.dump((embeddings, adjacency_matrix, vocab,nodes), f)



if (__name__ == '__main__'):
    makeGraph()