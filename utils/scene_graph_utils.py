import torch
import torch.nn as nn   
from utils.dataset_utils import get_pixel_distance
import networkx as nx
import matplotlib.pyplot as plt


def generate_SG_from_bboxs(bboxes, threshold):
    objects = []
    n = len(bboxes)
    adjacency_matrix = torch.zeros((n, n))
    for i in range(n):
        objects.append(bboxes[i]['object'])
        for j in range(i + 1, n):
            if get_pixel_distance(bboxes[i]['object_bbox'], bboxes[j]['object_bbox']) < threshold:
                adjacency_matrix[i][j] = 1
                adjacency_matrix[j][i] = 1
    return objects, adjacency_matrix    

def get_KG_active_idx(SG_nodes, SG_Adj, KG_vocab , obj):
    active_idx = []

    if 'saucepan' in SG_nodes:
        SG_nodes[SG_nodes.index('saucepan')] = 'pan'
    elif 'frying pan' in SG_nodes:
        SG_nodes[SG_nodes.index('frying pan')] = 'pan'
    elif 'small pan' in SG_nodes:
        SG_nodes[SG_nodes.index('small pan')] = 'pan'
    elif 'wok' in SG_nodes:
        SG_nodes[SG_nodes.index('wok')] = 'pan'
        
    if 'vegetable rice' in SG_nodes:
        SG_nodes[SG_nodes.index('vegetable rice')] = 'rice'
    if 'mug' in SG_nodes:
        SG_nodes[SG_nodes.index('mug')] = 'cup'
    if "minced meat" in SG_nodes:
        SG_nodes[SG_nodes.index("minced meat")] = "meat"

    # Find the index of the object in KG_vocab
    obj_idx = KG_vocab.index(obj)
    active_idx.append(obj_idx)

    # Find the neighbors of the object in SG_Adj
    neighbors = torch.nonzero(SG_Adj[SG_nodes.index(obj)])

    # Find the indices of neighbors in KG_vocab
    # neighbors_idx = [KG_vocab.index(SG_nodes[neighbor]) for neighbor in neighbors]
    
    for neighbor in neighbors:
        try :
            active_idx.append(KG_vocab.index(SG_nodes[neighbor]))
        except:
            continue
    
    # active_idx.extend(neighbors_idx)

    # Remove duplicates and return the result
    return torch.tensor(list(set(active_idx)))

def visualize_graph(nodes, adjacency_matrix):
    # Create a graph
    G = nx.Graph()

    # Add nodes to the graph
    G.add_nodes_from(nodes)

    # Add edges to the graph based on the adjacency matrix
    for i in range(len(nodes)):
        for j in range(len(nodes)):
            if adjacency_matrix[i][j] == 1:
                G.add_edge(nodes[i], nodes[j])

    # Draw the graph
    pos = nx.spring_layout(G)  # You can use other layouts as well
    nx.draw(G, pos, with_labels=True, font_weight='bold', node_size=700, node_color='skyblue', font_size=8)

    # Display the graph
    # plt.show()
    plt.savefig("results/scene_graph_inference.png")