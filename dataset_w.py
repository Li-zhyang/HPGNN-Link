from torch_geometric.data import Data, Dataset
import numpy as np
import torch
from torch_geometric.utils import k_hop_subgraph, dense_to_sparse, subgraph
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

class MyDataset_w(Dataset):
    def __init__(self, file, A, links, labels, k, class_num, device):
        super(MyDataset_w, self).__init__()
        self.A = A[0]
        self.edge_index, self.edge_type = dense_to_sparse(self.A)
        self.links = links
        self.labels = labels
        self.k = k

        self.class_values = torch.eye(class_num, dtype=torch.float64)
        self.device = device
        if not os.path.exists(file):
            self.data = []
            t=0
            for idx in tqdm(range(self.links.shape[1])):
                i, j = self.links[0][idx].item(), self.links[1][idx].item()
                g_label = self.labels[idx]
                tmp = subgraph_extraction_labeling(
                    (i, j), self.edge_index.clone(), self.edge_type.clone(), self.k, self.class_values, g_label
                )
                if tmp:
                    self.data.append(construct_pyg_graph(*tmp).to('cpu'))
                else:
                    t+=1
                    print(t)
            torch.save(self.data, file)
            print(t)

        self.data = torch.load(file)
        for idx in range(len(self.data)):
            self.data[idx] = self.data[idx]

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
    if not fringe:
        return set([])
    return set([i.item() for i in torch.nonzero(A[list(fringe)])[:,1]])
    
def subgraph_extraction_labeling(ind, edge_index, edge_type, k=1, class_values=None, y=1):
    remove_idx = torch.all(edge_index==torch.tensor([[ind[0]],[ind[1]]]),dim=0)
    if remove_idx.any():
        if edge_type[remove_idx]!=0:
            edge_index = edge_index[:, ~remove_idx]
            edge_type = edge_type[~remove_idx]

    subset, ssedge1, _, edge_mask = k_hop_subgraph(ind, k, edge_index, relabel_nodes=False)
    subset2, ssedge2, _, _ = k_hop_subgraph(ind, k, edge_index, relabel_nodes=False, flow="target_to_source")
    subset = torch.cat([subset,subset2]).unique()
    if len(subset)>2000 or len(subset)<=3:
        return False

    mapping = torch.searchsorted(subset, torch.tensor(ind))

    sub_edge_index, r = subgraph(subset, edge_index, edge_type, relabel_nodes=True)
    u,v = sub_edge_index

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
            
    return u, v, r, y, node_labels, max_node_label, link, mapping

def one_hot(idx, length):
    idx = np.array(idx)
    x = np.zeros([len(idx), length])
    x[np.arange(len(idx)), idx] = 1.0
    return x

def construct_pyg_graph(u, v, r, y, node_labels, max_node_label, link, mapping=None):
    u, v = torch.LongTensor(u), torch.LongTensor(v)
    r = torch.LongTensor(r)
    edge_index = torch.stack([u, v], 0)
    edge_type = r
    x = torch.FloatTensor(one_hot(node_labels, max_node_label+1))
    links = link
    num_nodes = x.shape[0]
    hyper_edge = to_line_graph(edge_index, num_nodes)
    w = 0
    if mapping!=None:
        data = MyData(x, edge_index=edge_index, edge_type=edge_type, links = links, y=y.unsqueeze(0), s=mapping[0], t=mapping[1], hyper_edge = hyper_edge, edge_gt_att = w)
    else:
        data = MyData(x, edge_index=edge_index, edge_type=edge_type, links = links, y=y.unsqueeze(0), hyper_edge = hyper_edge, edge_gt_att = w)
    return data 
