import torch
import torch.nn as nn
import torch.nn.functional as F
from graph_utils import unbatch_gemb
from tqdm import tqdm
import networkx as nx
from torch_geometric.nn import HypergraphConv


def hook_y(grad):
    print(grad)

Q=50

class HyperWeight(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, node_in_dim = 0, out_dim = 1, dropout = 0.5):
        super().__init__()
        self.in_dim = in_dim
        self.node_in_dim = node_in_dim
        if node_in_dim != 0:
            self.in_dim += node_in_dim
        
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.hyperconv1 = HypergraphConv(2*hidden_dim, 2*hidden_dim)
        self.hyperconv2 = HypergraphConv(2*hidden_dim, hidden_dim)
        self.lin = nn.Sequential(nn.Linear(2*hidden_dim,hidden_dim),
                                    nn.ReLU(),
                                    nn.Dropout(dropout),
                                    nn.Linear(hidden_dim, 1),
                                    nn.Sigmoid())
        
    def forward(self, edge_index, edge_rep, x, hyper_edge, prototype,
                node_feature = None):

        if node_feature != None:
            f_u, f_v = node_feature[edge_index[0]], node_feature[edge_index[1]]
            edge_feature = (f_u + f_v) / 2
            edge_rep = torch.cat([edge_rep, edge_feature], dim = 1)

        node_weight = self.lin(torch.cat([x,prototype],dim=-1))
        hyper_weight = node_weight[hyper_edge[1]]
        edge_rep = self.hyperconv1(edge_rep, hyper_edge, hyper_weight)
        edge_rep = torch.sigmoid(edge_rep)
        edge_rep = F.dropout(edge_rep, p = self.dropout, training = self.training)
        edge_rep = self.hyperconv2(edge_rep, hyper_edge, hyper_weight)
        edge_rep = torch.sigmoid(edge_rep)
        edge_rep = F.dropout(edge_rep, p = self.dropout, training = self.training)
        
        return edge_rep

class Generator(nn.Module):

    def __init__(self, nfeat, nhid, max_num_nodes=100, dropout=0.5, lr=0.01, weight_decay=5e-4, device=None, num_class=0):

        super(Generator, self).__init__()

        assert device is not None, "Please specify 'device'!"
        self.device = device
        self.nfeat = nfeat
        self.hidden_sizes = [nhid]
        self.dropout = dropout
        self.lr = lr
        self.max_num_nodes = max_num_nodes
        self.weight_decay = weight_decay
        self.hypergraph = HyperWeight(nfeat, nfeat)
        self.attn = nn.Parameter(torch.FloatTensor([0.5, 0.5]))
        self.lin = nn.Sequential(nn.Linear(3*nfeat,4*nfeat),
                                    nn.ReLU(),
                                    nn.Dropout(dropout),
                                    nn.Linear(4*nfeat,nfeat),
                                    nn.ReLU(),
                                    nn.Dropout(dropout))
        self.classifier = nn.Sequential(nn.Linear(2*nfeat,nfeat),
                                    nn.ReLU(),
                                    nn.Dropout(dropout),
                                    nn.Linear(nfeat,1))
        self.T_l=torch.tensor(0.3).to(device)
        self.num_class = num_class
    
    def forward(self, x, graph_emb, edge_index, edge_type, batch, hyper_edge):
        gemb = unbatch_gemb(edge_index, batch, graph_emb)
        col, row = edge_index
        f1, f2 = x[col], x[row]
        xij = torch.cat([f1, f2], dim=-1)
        proto = graph_emb[batch]
        sij = self.hypergraph(edge_index, xij, x, hyper_edge, proto)
        xij = self.lin(torch.cat([xij, gemb],dim=-1))
        sij = self.attn[0]*xij + self.attn[1]*sij
        sij = torch.cat([sij, gemb], dim=-1)
        sij = self.classifier(sij)
        if self.training:
            sij = self.extract_graph_t(sij)
        else:
            sij = torch.sigmoid(sij)
        return edge_index,edge_type,sij
    
    def extract_graph_t(self, sij):
        random_noise = torch.empty_like(sij).uniform_(1e-10, 1 - 1e-10)     #bern(puv)
        random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
        sij_hat = (sij + random_noise).sigmoid()    #gumble-softmax

        return sij_hat
    
    def find_dijkstra_path(self, edge_index, edge_type, edge_weight, s, t, num_nodes, bidirec = False):
        cutoff=3
        G = nx.DiGraph()
        G.add_nodes_from(range(num_nodes))
        edge_index = edge_index.transpose(0,1)

        for i in range(len(edge_index)):
            source, target = edge_index[i]
            etype = edge_type[i].item()
            weight = edge_weight[i].item()
            G.add_edge(source.item(), target.item(), weight=weight, edge_type=etype)

        G_undirected = G
        if bidirec:
            cutoff = 2
            G_undirected = nx.Graph()

            for source, target, data in G.edges(data=True):
                u = min(source, target)
                v = max(source, target)
                weight = data['weight']
                edge_type = data['edge_type']

                if G_undirected.has_edge(u, v):
                    existing_weight = G_undirected[u][v]['weight']
                    if weight > existing_weight:
                        G_undirected[u][v]['weight'] = weight
                        G_undirected[u][v]['edge_type'] = edge_type
                else:
                    G_undirected.add_edge(u, v, weight=weight, edge_type=edge_type)
        
        src = s.item()
        tgt = t.item()
        paths = nx.all_simple_paths(G_undirected, source=src, target=tgt, cutoff=cutoff)
        if len(list(paths))!=0:
            max_weight_path = None
            max_weight = float('-inf')
            paths = nx.all_simple_paths(G_undirected, source=src, target=tgt, cutoff=cutoff)
            for path in paths:
                path_weight = sum(G_undirected[path[i]][path[i + 1]]['weight'] for i in range(len(path) - 1))/(len(path)-1)

                if path_weight > max_weight:
                    max_weight = path_weight
                    max_weight_path = path
            
            edge_index_hat = []
            edge_type_hat = []
            sij_hat = []

            for i in range(len(max_weight_path) - 1):
                u = max_weight_path[i]
                v = max_weight_path[i + 1]
                weight = G_undirected[u][v]['weight']
                etype = G_undirected[u][v]['edge_type']

                if G.has_edge(u,v):
                    if G[u][v]['weight']==weight and G[u][v]['edge_type']==etype:
                        edge_index_hat.append([u, v])
                    else:
                        edge_index_hat.append([v, u])
                else:
                    edge_index_hat.append([v, u])
                    
                edge_type_hat.append(etype)
                sij_hat.append(weight)

            return torch.tensor(edge_index_hat).transpose(0,1).to(edge_index.device)
        else:
            return torch.tensor([[-1,-1],[-1,-1]]).transpose(0,1).to(edge_index.device)

    
    def get_explaination(self, x, graph_emb, edge_index, edge_type, s, t, hyper_edge, bidirec=False):
        col, row = edge_index
        f1, f2 = x[col], x[row]
        gemb = graph_emb.repeat(edge_index.shape[1],1)
        xij = torch.cat([f1,f2], dim=-1)
        proto = graph_emb.repeat(x.shape[0],1)
        sij = self.hypergraph(edge_index, xij, x, hyper_edge, proto)
        xij = self.lin(torch.cat([xij, gemb],dim=-1))
        sij = self.attn[0]*xij + self.attn[1]*sij
        sij = torch.cat([sij, gemb], dim=-1)
        sij = self.classifier(sij)
        sij = torch.sigmoid(sij)
        edge_index_hat = self.find_dijkstra_path(edge_index, edge_type, sij, s, t, x.shape[0], bidirec)
        
        return edge_index_hat, sij    
