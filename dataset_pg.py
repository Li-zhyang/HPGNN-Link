from torch_geometric.data import Data, Dataset
import numpy as np
import torch
import dgl
from utils import hetero_src_tgt_khop_in_subgraph
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


class MyDataset_pg(Dataset):
    def __init__(self, file, g, links, k, max_degree, class_num, exp, exp_path, device, k_core, pred_type):
        super(MyDataset_pg, self).__init__()
        self.g = g
        self.links = torch.cat([links[0].unsqueeze(0),links[2].unsqueeze(0)],dim=0)
        self.labels = links[1]
        self.k = k
        self.k_core = k_core
        self.max_degree = max_degree

        self.class_values = torch.eye(class_num, dtype=torch.float64)
        self.exp = exp
        self.device = device
        self.exp_path = exp_path
        self.pred_type = pred_type
        t = 0
        if not os.path.exists(file):
            self.data = []
            
            for idx in tqdm(range(self.links.shape[1])):
                i, j = self.links[0][idx].item(), self.links[1][idx].item()
                g_label = self.labels[idx].item()
                exp = self.exp[(i,j)] if (i,j) in self.exp else torch.empty((3,0), dtype=torch.int64)
                exp_path = self.exp_path[(i,j)] if (i,j) in self.exp else torch.empty((3,0), dtype=torch.int64)
                tmp = subgraph_extraction_labeling(
                    (i, j), self.g, exp, exp_path, self.k, self.max_degree, 
                    self.class_values, g_label, self.k_core, self.pred_type
                )
                if tmp:
                    self.data.append(construct_pyg_graph(*tmp).to('cpu'))
                else:
                    t+=1
            print(t)
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
    
def remove_edges_of_high_degree_nodes(ghomo, max_degree=10, always_preserve=[]):
    '''
    For all the nodes with degree higher than `max_degree`, 
    except nodes in `always_preserve`, remove their edges. 
    
    Parameters
    ----------
    ghomo : dgl homogeneous graph
    
    max_degree : int
    
    always_preserve : iterable
        These nodes won't be pruned.
    
    Returns
    -------
    low_degree_ghomo : dgl homogeneous graph
        Pruned graph with edges of high degree nodes removed

    '''
    d = ghomo.in_degrees()
    high_degree_mask = d > max_degree
    
    # preserve nodes
    high_degree_mask[always_preserve] = False    

    high_degree_nids = ghomo.nodes()[high_degree_mask]
    u, v = ghomo.edges()
    high_degree_edge_mask = torch.isin(u, high_degree_nids) | torch.isin(v, high_degree_nids)
    high_degree_u, high_degree_v = u[high_degree_edge_mask], v[high_degree_edge_mask]
    high_degree_eids = ghomo.edge_ids(high_degree_u, high_degree_v)
    low_degree_ghomo = dgl.remove_edges(ghomo, high_degree_eids)
    
    return low_degree_ghomo

def remove_edges_except_k_core_graph(ghomo, k, always_preserve=[]):
    '''
    Find the `k`-core of `ghomo`.
    Only isolate the low degree nodes by removing theirs edges
    instead of removing the nodes, so node ids can be kept.
    
    Parameters
    ----------
    ghomo : dgl homogeneous graph
    
    k : int
    
    always_preserve : iterable
        These nodes won't be pruned.
    
    Returns
    -------
    k_core_ghomo : dgl homogeneous graph
        The k-core graph
    '''
    k_core_ghomo = ghomo
    degrees = k_core_ghomo.in_degrees()
    k_core_mask = (degrees > 0) & (degrees < k)
    k_core_mask[always_preserve] = False
    u, v = k_core_ghomo.edges()
    k_core_mask[u[torch.isin(v, torch.tensor(always_preserve))]] = False
    k_core_mask[v[torch.isin(u, torch.tensor(always_preserve))]] = False
    
    while k_core_mask.any():
        k_core_nids = k_core_ghomo.nodes()[k_core_mask]
        
        u, v = k_core_ghomo.edges()
        k_core_edge_mask = torch.isin(u, k_core_nids) | torch.isin(v, k_core_nids)
        k_core_u, k_core_v = u[k_core_edge_mask], v[k_core_edge_mask]
        k_core_eids = k_core_ghomo.edge_ids(k_core_u, k_core_v)

        k_core_ghomo = dgl.remove_edges(k_core_ghomo, k_core_eids)
        
        degrees = k_core_ghomo.in_degrees()
        k_core_mask = (degrees > 0) & (degrees < k)
        k_core_mask[always_preserve] = False
        u, v = k_core_ghomo.edges()
        k_core_mask[u[torch.isin(v, torch.tensor(always_preserve))]] = False
        k_core_mask[v[torch.isin(u, torch.tensor(always_preserve))]] = False

    return k_core_ghomo
    
def prune_graph(ghetero, prune_max_degree=-1, k_core=3, always_preserve=[]):
    # Prune edges by (optionally) removing edges of high degree nodes and extracting k-core
    # The pruning is computed on the homogeneous graph, i.e., ignoring node/edge types
    ghomo = dgl.to_homogeneous(ghetero)
    device = ghetero.device
    ghomo.edata['eid_before_prune'] = torch.arange(ghomo.num_edges()).to(device)
    
    if prune_max_degree > 0:
        max_degree_pruned_ghomo = remove_edges_of_high_degree_nodes(ghomo, prune_max_degree, always_preserve)
        k_core_ghomo = remove_edges_except_k_core_graph(max_degree_pruned_ghomo, k_core, always_preserve)
        # if k_core_ghomo.num_edges()>10000:
        #     print('a')
        
        if k_core_ghomo.num_edges() <= 0: # no k-core found
            pruned_ghomo = max_degree_pruned_ghomo
        else:
            pruned_ghomo = k_core_ghomo
    else:
        k_core_ghomo = remove_edges_except_k_core_graph(ghomo, k_core, always_preserve)
        if k_core_ghomo.num_edges() <= 0: # no k-core found
            pruned_ghomo = ghomo
        else:
            pruned_ghomo = k_core_ghomo

    d_in = pruned_ghomo.in_degrees()
    d_out = pruned_ghomo.out_degrees()
    zero_degree_mask = (d_in != 0) | (d_out != 0)
    zero_degree_mask[always_preserve] = True    
    zero_degree_nids = pruned_ghomo.nodes()[zero_degree_mask]
    
    pruned_ghomo_eids = pruned_ghomo.edata['eid_before_prune']
    pruned_ghomo_eid_mask = torch.zeros(ghomo.num_edges()).bool()
    pruned_ghomo_eid_mask[pruned_ghomo_eids] = True

    # Apply the pruning result on the heterogeneous graph
    etypes_to_pruned_ghetero_eid_masks = {}
    pruned_ghetero = ghetero
    cum_num_edges = 0
    for etype in ghetero.canonical_etypes:
        num_edges = ghetero.num_edges(etype=etype)
        pruned_ghetero_eid_mask = pruned_ghomo_eid_mask[cum_num_edges:cum_num_edges+num_edges]
        etypes_to_pruned_ghetero_eid_masks[etype] = pruned_ghetero_eid_mask

        remove_ghetero_eids = (~ pruned_ghetero_eid_mask).nonzero().view(-1).to(device)
        pruned_ghetero = dgl.remove_edges(pruned_ghetero, eids=remove_ghetero_eids, etype=etype)

        cum_num_edges += num_edges

    
    pruned_ghetero_new = dgl.node_subgraph(pruned_ghetero, zero_degree_nids, relabel_nodes=True, store_ids=True)
            
    return pruned_ghetero_new, etypes_to_pruned_ghetero_eid_masks
    
    
def subgraph_extraction_labeling(ind, g, exp, exp_path, k=1, max_degree=None, class_values=None, y=1, k_core=3, pred_type = 1):
    if y!=0 and y!=pred_type:   #syn5,aug1
        eid = g.edge_ids(ind[0], ind[1], etype=str(y))
        new_g = dgl.remove_edges(g, eid, etype=str(y))
    else:
        new_g = g
    
    (comp_g_src_nid, 
     comp_g_tgt_nid, 
     comp_g, 
     comp_g_feat_nids) = hetero_src_tgt_khop_in_subgraph(ind[0], ind[1], new_g, k)
    
    comp_src,comp_tgt,comp_r = torch.empty(0, dtype=torch.int64),torch.empty(0, dtype=torch.int64),torch.empty(0, dtype=torch.int64)

    for etype in comp_g.etypes:
        u, v = comp_g.edges(etype=etype)
        comp_src = torch.cat([comp_src, u])
        comp_tgt = torch.cat([comp_tgt, v])
        comp_r = torch.cat([comp_r,torch.tensor([int(etype)]*u.shape[0],dtype = torch.int64)])
    
    if comp_g.num_edges()==0:
        comp_g.add_edges(comp_g_tgt_nid.item(), comp_g_src_nid.item(), etype=('no_type','0','no_type'))
    
    ml_ghetero, etypes_to_pruned_ghetero_eid_masks = prune_graph(comp_g, max_degree, k_core, [comp_g_src_nid.item(), comp_g_tgt_nid.item()])

    comp2prune = dict(zip(ml_ghetero.ndata[dgl.NID].numpy(), range(0, ml_ghetero.num_nodes())))
    if ml_ghetero.num_nodes()>1000:
        if y==1:
            print('a')
        return False

    if ml_ghetero.num_edges()==0:
        ml_ghetero = dgl.add_edges(ml_ghetero, torch.tensor([comp2prune[comp_g_tgt_nid.item()]]), torch.tensor([comp2prune[comp_g_src_nid.item()]]), etype=str(y))

    src,tgt,r = torch.empty(0, dtype=torch.int64),torch.empty(0, dtype=torch.int64),torch.empty(0, dtype=torch.int64)

    for etype in ml_ghetero.etypes:
        u, v = ml_ghetero.edges(etype=etype)
        src = torch.cat([src, u])
        tgt = torch.cat([tgt, v])
        r = torch.cat([r,torch.tensor([int(etype)]*u.shape[0],dtype = torch.int64)])

    y = class_values[y]

    node_labels = torch.zeros(ml_ghetero.num_nodes(),dtype=torch.int64)+2
    for i in range(k-1,0,-1):
        pred_dict = {'no_type': torch.tensor([comp2prune[comp_g_src_nid.item()], comp2prune[comp_g_tgt_nid.item()]])}
        i_hop = dgl.khop_in_subgraph(ml_ghetero, pred_dict, i)[0]
        node_labels[i_hop.ndata[dgl.NID]]=i
    node_labels[[comp2prune[comp_g_src_nid.item()], comp2prune[comp_g_tgt_nid.item()]]]=0
    max_node_label = k

    link = torch.zeros((ml_ghetero.num_nodes(),1),dtype=torch.int64)
    link[comp2prune[comp_g_src_nid.item()]]=1
    link[comp2prune[comp_g_tgt_nid.item()]]=2
    mapping = (comp2prune[comp_g_src_nid.item()], comp2prune[comp_g_tgt_nid.item()])

    ent2idx = dict(zip(comp_g_feat_nids.numpy(), range(0,comp_g_feat_nids.shape[0])))
    tu, tr, tv = exp
    new_tu = torch.tensor([comp2prune[ent2idx[i.item()]] for i in tu],dtype = torch.int64)
    new_tv = torch.tensor([comp2prune[ent2idx[i.item()]] for i in tv],dtype = torch.int64)
    traces_edge_index = torch.stack([new_tu,new_tv], 0)
    w = torch.where((torch.stack([src,tgt]).T[:, None] == traces_edge_index.T).all(dim=2).any(dim=1), 1.0, 0.0)

    tu, trace_path_type, tv = exp_path
    new_tu = torch.tensor([comp2prune[ent2idx[i.item()]] for i in tu],dtype = torch.int64)
    new_tv = torch.tensor([comp2prune[ent2idx[i.item()]] for i in tv],dtype = torch.int64)
    trace_path_index = torch.stack([new_tu,new_tv], 0)

    comp_num = 0
    for t in tr.unique():
        comp_num += comp_g.num_edges(etype = str(t.item()))-ml_ghetero.num_edges(etype = str(t.item()))
    return src, tgt, r, y, node_labels, max_node_label, link, traces_edge_index, tr, comp_g_feat_nids, comp_src, comp_tgt, trace_path_type, comp_num, mapping, w

def one_hot(idx, length):
    idx = np.array(idx)
    x = np.zeros([len(idx), length])
    x[np.arange(len(idx)), idx] = 1.0
    return x

def construct_pyg_graph(u, v, r, y, node_labels, max_node_label, link, trace_edge_index, trace_edge_type, subset, comp_src, comp_tgt, trace_path_type, comp_num, mapping, w=None):
    u, v = torch.LongTensor(u), torch.LongTensor(v)
    r = torch.LongTensor(r)
    trace_edge_type = torch.LongTensor(trace_edge_type)
    trace_path_type = torch.LongTensor(trace_path_type)
    edge_index = torch.stack([u, v], 0)
    edge_type = r
    x = torch.FloatTensor(one_hot(node_labels, max_node_label+1))
    links = link
    comp_src, comp_tgt = torch.LongTensor(comp_src), torch.LongTensor(comp_tgt)
    num_nodes = x.shape[0]
    hyper_edge = to_line_graph(edge_index, num_nodes)
    data = MyData(x, edge_index=edge_index, edge_type=edge_type, links = links, y=y.unsqueeze(0), trace_edge_index=trace_edge_index, trace_edge_type=trace_edge_type, comp_num=comp_num, s = mapping[0], t = mapping[1], subset = subset.unsqueeze(1), hyper_edge = hyper_edge, edge_gt_att = w)
    return data 
