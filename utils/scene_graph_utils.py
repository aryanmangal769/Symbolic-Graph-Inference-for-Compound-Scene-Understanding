import torch
import torch.nn as nn   
from utils.dataset_utils import get_pixel_distance
import networkx as nx
import matplotlib.pyplot as plt
from torchvision import transforms

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

def generate_SG(bboxes, threshold, img, model_vit):
    transform = transforms.Compose([ transforms.Resize((224, 224)), transforms.ToTensor(),])

    objects = []
    embeddings = []
    n = len(bboxes)
    adjacency_matrix = torch.zeros((n, n))
    for i in range(n):
        objects.append(bboxes[i]['object'])
        embeddings.append(model_vit(transform(img.crop(bboxes[i]['object_bbox'])).unsqueeze(0)).detach())
        for j in range(i + 1, n):
            if get_pixel_distance(bboxes[i]['object_bbox'], bboxes[j]['object_bbox']) < threshold:
                adjacency_matrix[i][j] = 1
                adjacency_matrix[j][i] = 1

    embeddings = torch.cat(embeddings, dim=0)
    return objects, embeddings, adjacency_matrix   

def get_KG_active_idx(SG_nodes, SG_Adj, KG_vocab , obj):
    active_idx = []
    SG_nodes = [node.split('_')[0] for node in SG_nodes]

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
    try :
        obj_idx = KG_vocab.index(obj)
        active_idx.append(obj_idx)
    except:
        ## Appending with background such that the model can still run
        active_idx.append(1)

    # Find the neighbors of the object in SG_Adj
    if obj not in SG_nodes:  # If the object is not present in the scene graph we takes the first object as the principal object
        obj = SG_nodes[0]
        # print("Object not in scene graph, taking the first object at the principal object")
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

def get_SG_active_idx(SG_nodes, SG_Adj, KG_vocab , obj):
    active_idx = []
    active_nodes = []
    SG_nodes = [node.split('_')[0] for node in SG_nodes]
    SG_Adj_neighbours = SG_Adj.clone()

    # Find the index of the object in KG_vocab
    # obj_idx = KG_vocab.index(obj)

    # Find the neighbors of the object in SG_Adj
    if obj not in SG_nodes:  # If the object is not present in the scene graph we takes the first object as the principal object
        obj = SG_nodes[0]
        # print("Object not in scene graph, taking the first object at the principal object")
    
    active_idx.append(SG_nodes.index(obj))
    if obj in get_neighbour_nodes(SG_Adj, SG_nodes,SG_nodes.index(obj)):
        SG_Adj_neighbours[SG_nodes.index(obj), SG_nodes.index(obj)] = 1

    neighbors = torch.nonzero(SG_Adj[SG_nodes.index(obj)])

    # Find the indices of neighbors in KG_vocab
    # neighbors_idx = [KG_vocab.index(SG_nodes[neighbor]) for neighbor in neighbors]
    
    for neighbor in neighbors:
        try :
            if SG_nodes[neighbor] not in active_nodes:
                active_idx.append(neighbor)
                if SG_nodes[neighbor] in get_neighbour_nodes(SG_Adj,SG_nodes, neighbor):
                    SG_Adj_neighbours[neighbor, neighbor] = 1
                active_nodes.append(SG_nodes[neighbor])
        except:
            continue
    
    # active_idx.extend(neighbors_idx)

    # Remove duplicates and return the result
    return torch.tensor(active_idx)

def update_SG_adj(SG_nodes, SG_Adj, KG_vocab , obj):
    active_idx = []
    active_nodes = []
    SG_nodes = [node.split('_')[0] for node in SG_nodes]
    SG_Adj_neighbours = SG_Adj.clone()

    # Find the index of the object in KG_vocab
    # obj_idx = KG_vocab.index(obj)

    # Find the neighbors of the object in SG_Adj
    if obj not in SG_nodes:  # If the object is not present in the scene graph we takes the first object as the principal object
        obj = SG_nodes[0]
        # print("Object not in scene graph, taking the first object at the principal object")
    
    active_idx.append(SG_nodes.index(obj))
    if obj in get_neighbour_nodes(SG_Adj, SG_nodes,SG_nodes.index(obj)):
        SG_Adj_neighbours[SG_nodes.index(obj), SG_nodes.index(obj)] = 1

    neighbors = torch.nonzero(SG_Adj[SG_nodes.index(obj)])

    # Find the indices of neighbors in KG_vocab
    # neighbors_idx = [KG_vocab.index(SG_nodes[neighbor]) for neighbor in neighbors]
    
    for neighbor in neighbors:
        try :
            if SG_nodes[neighbor] not in active_nodes:
                active_idx.append(neighbor)
                if SG_nodes[neighbor] in get_neighbour_nodes(SG_Adj,SG_nodes, neighbor):
                    SG_Adj_neighbours[neighbor, neighbor] = 1
                active_nodes.append(SG_nodes[neighbor])
        except:
            continue
    
    # active_idx.extend(neighbors_idx)

    # Remove duplicates and return the result
    return SG_Adj_neighbours


def get_neighbour_nodes(adj, nodes, node):
    return [nodes[i] for i in torch.nonzero(adj[node])]

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