import torch
import numpy as np
import pickle
from torchtext.vocab import GloVe
from typing import List, Tuple
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

def getKitchenRelation_automatic(dataset_path):
    base_dir = '/data/aryan/Seekg/Datasets/epic_kitchens_affordances/data'
    epic_kitchens = EPIC_Kitchens(base_dir, None)
    paths = []
    objects = []
    verbs = []
    dataset = pickle.load(open(dataset_path, 'rb'))

    # Create a dictionary to store objects for each verb
    object_verb_mapping = {}

    for obj in dataset:
        for verb in dataset[obj]:
            for path in dataset[obj][verb]:
                paths.append(path)
                objects.append(obj)
                verbs.append(verb)

                kitchen, sample_id = path.split('/')
                subset = sample_id.split('_')[0] + '_' + sample_id.split('_')[1]
                VISOR_bboxs, VISOR_active_objects_list = self.VISOR_bbox(sample_id + '.jpg', sequence)

                # Create a dictionary to store active objects for each verb
                if obj not in object_verb_mapping:
                    object_verb_mapping[obj] = {}

                # Create a set for the current verb
                if verb not in object_verb_mapping[obj]:
                    object_verb_mapping[obj][verb] = []

                # Add active objects to the set for the current verb
                object_verb_mapping[obj][verb].append(VISOR_active_objects_list)

    relations = []
    affordances = []
    tools = []

    # Print or process the resulting dictionary as needed
    for obj, verb_dict in object_verb_mapping.items():
        print(f"Object: {obj}")
        for verb, active_objects_list in verb_dict.items():
            affordances.append([obj, verb])
            print(f"  Verb: {verb}, Active Objects: {active_objects_list}")
            top_active_objects = Counter(active_objects_list).most_common(5)[0]
            for item in top_active_objects:
                tools.append([verb, item])
                relations.append([obj, item])



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

    with open('/data/aryan/Seekg/Datasets/epic_kitchens_affordances/data/epic_kitchens_KG.pkl', 'wb') as f:
        pickle.dump((embeddings, adjacency_matrix, vocab,nodes), f)



if (__name__ == '__main__'):
    makeGraph()