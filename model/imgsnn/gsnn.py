import torch
import torch.nn as nn
from model.graph_models.gcn import GCN
from model.graph_models.gcn_image import GCNI
from model.imgsnn.importance_net import Importance_net
from utils.graph_utils import get_neighbour_nodes, merge_graphs
import torch.nn.functional as F
import pdb



class IMGSNN(nn.Module):
    '''
    GSNN: Graph Search Neural Network
    '''

    def __init__(self, config, KG_vocab, KG_nodes):
        super().__init__()
        self.config = config
        self.KG_vocab = KG_vocab
        self.KG_nodes = KG_nodes

        nfeat = config['gcn']['n_feat']
        nhid = config['gcn']['n_hid']
        nout = config['gcn']['n_out']
        dropout = config['gcn']['dropout']
        n_layers = config['gcn']['n_layers']
        self.image_conditioning = config['gcn']['image_conditioning']
        self.n_steps = config['gsnn']['n_steps']
        self.step_threshold = config['gsnn']['step_threshold']
        

        self.gcns = nn.ModuleList([GCN(nfeat, nhid, nout, n_layers, dropout) for _ in range(self.n_steps)])
        self.imps = nn.ModuleList([Importance_net(nout) for _ in range(self.n_steps-1)])
        self.imps.append(Importance_net(nout, self.image_conditioning, config['vit']['img_dim'], config['vit']['num_classes']))

        self._avg_steps = 0
        self.SG_net = nn.Linear(768, nfeat)

        # self.gcn = GCN(nfeat, nhid, nout,n_layers, dropout)
        # self.imp = Importance_net(nout)

    def forward(self, x, KG_adj, SG_adj,SG_embeddings, active_idx_init, img_feat):
        '''
        x: Knowledge graph embedding
        adj: Adjacency matrix of the knowledge graph
        active_idx: Active node index
        '''
        SG_embeddings= self.SG_net(SG_embeddings)
        SG_embeddings =  SG_adj@SG_embeddings

        x, adj = merge_graphs(x, KG_adj, SG_adj,SG_embeddings, active_idx_init)

        active_idx = active_idx_init[0]
        
        for step in range(self.n_steps -1):
            neighbor_idx = get_neighbour_nodes(adj, active_idx)
            # for node in active_idx:
            #     print(self.KG_vocab[node])

            # print("#####")
            neighbor_idx = [node for node in neighbor_idx if node < KG_adj.shape[0]]
            neighbor_idx = torch.tensor([node for node in neighbor_idx if self.KG_vocab[node] in self.KG_nodes['objects'][0]]).to(x.device)

            # for node in neighbor_idx:
            #     print(self.KG_vocab[node])
            # print("#####")

            current_idx = torch.cat(( active_idx, neighbor_idx)) 
            # print(current_idx)

            h = self.gcns[step](x[current_idx, :].clone(), adj[current_idx][:, current_idx].clone())

            imp = self.imps[step](h, adj[current_idx][:,current_idx])

            if self.step_threshold *torch.max(imp[:len(active_idx)]) > torch.topk(imp[len(active_idx):], 1).values[0]:
                break

            # self._avg_steps += 1
            # print(self._avg_steps)
            try:
                active_idx = torch.cat((active_idx, neighbor_idx[torch.topk(imp[len(active_idx):], 1).indices]))
                # print(self.KG_vocab[neighbor_idx[torch.topk(imp[len(active_idx):], 1).indices][0]])
            except:
                continue

            # print("#####")
            # print(active_idx)


        neighbor_idx = get_neighbour_nodes(adj, active_idx)
        neighbor_idx = [node for node in neighbor_idx if node < KG_adj.shape[0]]
        neighbor_idx = torch.tensor([node for node in neighbor_idx if self.KG_vocab[node] not in self.KG_nodes['objects'][0]]).to(x.device)
        # print(torch.sum(x, dim=1))
        current_idx = torch.cat(( active_idx, neighbor_idx)) 

        h = self.gcns[-1](x[current_idx, :].clone(), adj[current_idx][:, current_idx].clone())       
        # print(torch.sum(h, dim=1))

        imp = self.imps[-1](h, adj[current_idx][:,current_idx], img_feat)

        return imp, current_idx[len(active_idx):]
