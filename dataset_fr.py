from torch_geometric.data import Data, Dataset
import numpy as np
import torch
import scipy.sparse as ssp
from torch_geometric.utils import k_hop_subgraph, dense_to_sparse
import os
from tqdm import tqdm

def to_line_graph(edge_index, num_nodes):
    hyperedges = [[] for i in range(num_nodes)]
    num_edges = edge_index.shape[1]
    for i in range(num_edges):
        u, v = edge_index.T[i]
        hyperedges[u].append(i)
        hyperedges[v].append(i)
    
    hyperedge_index = []
    for i in range(num_nodes):
        hyperedge = torch.tensor(hyperedges[i], dtype = torch.long,
                                    device = edge_index.device)
        index = torch.empty_like(hyperedge).fill_(i).to(edge_index.device)
        hyperedge_index.append(torch.stack([hyperedge, index]))
    return torch.cat(hyperedge_index, dim = 1)

class MyData(Data):
    def __cat_dim__(self, key, value, *args, **kwargs):
        if key == 'hyper_edge':
            return 1
        return super().__cat_dim__(key, value, *args, **kwargs)

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'hyper_edge':
            return torch.tensor([[self.edge_index.size(1)], [self.x.size(0)]])
        return super().__inc__(key, value, *args, **kwargs)

class MyDataset_fr(Dataset):
    def __init__(self, file, A, links, labels, k, class_num, traces, weights, device, unk_idx):
        super(MyDataset_fr, self).__init__()
        self.A = A[0]
        self.links = links
        self.labels = labels
        self.k = k
        self.unk_idx = unk_idx
        self.class_values = torch.eye(class_num, dtype=torch.float64)
        self.traces = traces
        self.weights = weights
        self.device = device
        if not os.path.exists(file):
            self.data = []

            for idx in tqdm(range(self.links.shape[1])):
                i, j = self.links[0][idx].item(), self.links[1][idx].item()
                g_label = self.labels[idx]
                tmp = subgraph_extraction_labeling(
                    (i, j), self.A.clone(), self.traces[idx], self.weights[idx], self.k, self.class_values, g_label, unk_idx
                )
                if tmp:
                    self.data.append(construct_pyg_graph(*tmp).to('cpu'))
            torch.save(self.data, file)

        self.data = torch.load(file)
        for idx in range(len(self.data)):
            self.data[idx] = self.data[idx].to(self.device)

    def get_by_label(self, y):
        label = torch.tensor([data.y.argmax().item() for data in self.data])
        idx = torch.where(label==y)[0]
        return self.index_select(idx)

    def len(self):
        return len(self.data)

    def edge_features(self):
        return len(set(self.class_values))

    def get(self, idx):
        return self.data[idx]
    
def neighbors(fringe, A):
    # find all 1-hop neighbors of nodes in fringe from A
    if not fringe:
        return set([])
    return set([i.item() for i in torch.nonzero(A[list(fringe)])[:,1]])
    
def subgraph_extraction_labeling(ind, A, traces, weights, k=1, class_values=None, y=1, unk_idx=2118):

    if ind[0]==5510 and ind[1]==4824:
        print('a')
    A[ind]=0
    edge_index,_ = dense_to_sparse(A)
    subset, _, _, edge_mask = k_hop_subgraph(ind, k, edge_index, num_nodes=A.shape[0], relabel_nodes=False)
    subset2, _, _, _ = k_hop_subgraph(ind, k, edge_index, num_nodes=A.shape[0], relabel_nodes=False, flow="target_to_source")
    subset = torch.cat([subset,subset2]).unique()
    subset = torch.cat([subset,torch.tensor([unk_idx])]).unique()

    mapping = torch.searchsorted(subset, torch.tensor(ind))
    subgraph = A[subset][:, subset]
    u, v, r = ssp.find(subgraph)

    tracesgraph = torch.zeros(subgraph.shape, dtype=torch.int64)
    weightsgraph = torch.zeros(subgraph.shape, dtype=torch.float32)

    tu = torch.searchsorted(subset, traces[0])
    tv = torch.searchsorted(subset, traces[2])
    tracesgraph[tu,tv] = traces[1]
    weightsgraph[tu,tv] = weights
    w = torch.where(weightsgraph[u,v]!=0, torch.sigmoid((weightsgraph[u,v]-0.2)/0.2), 0.0)

    traces_edge_index = torch.stack([tu,tv], 0)

    sub_edge_index,_ = dense_to_sparse(subgraph)
    node_labels = torch.zeros(len(subset),dtype=torch.int64)
    for i in range(k,0,-1):
        ssubset, _,_,_ = k_hop_subgraph(mapping, i, sub_edge_index, num_nodes=subset.shape[0], relabel_nodes=False)
        node_labels[ssubset]=i
        ssubset, _,_,_ = k_hop_subgraph(mapping, i, sub_edge_index, num_nodes=subset.shape[0], relabel_nodes=False, flow="target_to_source")
        node_labels[ssubset]=i
    node_labels[mapping]=0

    max_node_label = k

    y = class_values[y]

    link = torch.zeros((len(subset),1),dtype=torch.int64)
    link[mapping[0]]=1
    link[mapping[1]]=2
            
    return u, v, r, y, node_labels, max_node_label, link, ind, traces_edge_index, traces[1], weights.squeeze(0), mapping, subset, w

def one_hot(idx, length):
    idx = np.array(idx)
    x = np.zeros([len(idx), length])
    x[np.arange(len(idx)), idx] = 1.0
    return x

def construct_pyg_graph(u, v, r, y, node_labels, max_node_label, link, ind, trace_edge_index, trace_edge_type, trace_weight, mapping=None, subset=None, w=None):
    u, v = torch.LongTensor(u), torch.LongTensor(v)
    r = torch.LongTensor(r)
    edge_index = torch.stack([u, v], 0)
    edge_type = r
    x = torch.FloatTensor(one_hot(node_labels, max_node_label+1))
    links = link
    num_nodes = x.shape[0]
    hyper_edge = to_line_graph(edge_index, num_nodes)
    if mapping!=None:
        data = MyData(x, edge_index=edge_index, edge_type=edge_type, links = links, y=y.unsqueeze(0), trace_edge_index=trace_edge_index, trace_edge_type=trace_edge_type, trace_weight=trace_weight, s=mapping[0], t=mapping[1], ind = ind, subset = subset.unsqueeze(1), hyper_edge = hyper_edge, edge_gt_att = w)
    else:
        data = MyData(x, edge_index=edge_index, edge_type=edge_type, links = links, y=y.unsqueeze(0), trace_edge_index=trace_edge_index, trace_edge_type=trace_edge_type, trace_weight=trace_weight, ind = ind, subset = subset.unsqueeze(1), hyper_edge = hyper_edge, edge_gt_att = w)
    return data 
