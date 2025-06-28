import dgl
import torch
import numpy as np
from utils import eids_split
from torch_geometric.utils import negative_sampling

np.random.seed(123)

def process_data(g, 
                 val_ratio, 
                 test_ratio,
                 pred_etype = 'likes', graph_saving_path='datasets/synthetic'):
    '''
    Parameters
    ----------
    g : dgl graph
    
    val_ratio : float
    
    test_ratio : float
    
    src_ntype: string
        source node type
    
    tgt_ntype: string
        target node type

    neg: string
        One of ['pred_etype_neg', 'src_tgt_neg'], different negative sampling modes. See below.
    
    Returns
    ----------
    mp_g: 
        graph for message passing.
    
    graphs containing positive edges and negative edges for train, valid, and test
    '''
    C = 0
    N = g.num_nodes()-1
    src,tgt,r = torch.empty(0, dtype=torch.int64),torch.empty(0, dtype=torch.int64),torch.empty(0, dtype=torch.int64)

    for etype in g.etypes:
        if etype=='0':
            continue
        u, v = g.edges(etype=etype)
        num_samples = min(g.num_edges('1'), u.shape[0])
        C += num_samples
        idx = np.random.choice(range(u.shape[0]), num_samples, replace=False)
        u, v = u[idx], v[idx]
        src = torch.cat([src, u])
        tgt = torch.cat([tgt, v])
        r = torch.cat([r,torch.tensor([int(etype)]*u.shape[0],dtype = torch.int64)])

    C = int(C/(len(g.etypes)-1))

    eids = torch.arange(src.shape[0])
    train_pos_eids, val_pos_eids, test_pos_eids = eids_split(eids, val_ratio, test_ratio)

    train_pos_u, train_pos_v, train_pos_r = src[train_pos_eids], tgt[train_pos_eids], r[train_pos_eids]
    val_pos_u, val_pos_v = src[val_pos_eids], tgt[val_pos_eids]
    test_pos_u, test_pos_v, test_pos_r = src[test_pos_eids], tgt[test_pos_eids], r[test_pos_eids]


    # Edges not connecting src and tgt as negative edges
    # Collect all edges between the src and tgt
    src_tgt_indices = []
    for etype in g.etypes:
        adj = g.adj(etype=etype)
        src_tgt_index = adj.coalesce().indices()        
        src_tgt_indices += [src_tgt_index]
    src_tgt_u, src_tgt_v = torch.cat(src_tgt_indices, dim=1)

    all_edge_u = torch.cat([g.edges(etype = etype)[0] for etype in g.etypes],dim=0).unsqueeze(0)
    all_edge_v = torch.cat([g.edges(etype = etype)[1] for etype in g.etypes],dim=0).unsqueeze(0)
    neg_edges = negative_sampling(torch.cat([all_edge_u,all_edge_v],dim=0), num_nodes=N, num_neg_samples=C)

        
    neg_eids = torch.arange(neg_edges.shape[1])
    train_neg_eids, val_neg_eids, test_neg_eids = eids_split(neg_eids, val_ratio, test_ratio)
    neg_u, neg_v = neg_edges

    train_neg_u, train_neg_v = neg_u[train_neg_eids], neg_v[train_neg_eids]
    val_neg_u, val_neg_v = neg_u[val_neg_eids], neg_v[val_neg_eids]
    test_neg_u, test_neg_v = neg_u[test_neg_eids], neg_v[test_neg_eids]

    # Avoid losing dimension in single number slicing
    train_neg_u, train_neg_v = np.take(neg_u, train_neg_eids), np.take(neg_v, train_neg_eids)
    val_neg_u, val_neg_v = np.take(neg_u, val_neg_eids),np.take(neg_v, val_neg_eids)
    test_neg_u, test_neg_v = np.take(neg_u, test_neg_eids), np.take(neg_v, test_neg_eids)

    train_u = torch.cat([train_neg_u,train_pos_u]).unsqueeze(0)
    train_v = torch.cat([train_neg_v,train_pos_v]).unsqueeze(0)
    train_label = torch.cat([torch.zeros(train_neg_u.shape[0], dtype=torch.int64),train_pos_r]).unsqueeze(0)
    torch.save(torch.cat([train_u,train_label,train_v],dim=0), f'{graph_saving_path}_train_link')
    test_u = torch.cat([test_pos_u,test_neg_u]).unsqueeze(0)
    test_v = torch.cat([test_pos_v,test_neg_v]).unsqueeze(0)
    test_label = torch.cat([test_pos_r,torch.zeros(test_neg_u.shape[0], dtype=torch.int64)]).unsqueeze(0)
    torch.save(torch.cat([test_u,test_label,test_v],dim=0), f'{graph_saving_path}_test_link')
    return 

def get_nodehomo_graph(g, pred, pred_path):
    new_node_type = 'no_type'
    edge_dict = {}
    pred_dict = {}

    rel2idx = dict(zip(g.etypes, range(1,len(g.etypes)+1)))

    for etype in g.canonical_etypes:
        u, v = g.edges(etype = etype)
        new_u = g.ndata['nx_id'][etype[0]][u]
        new_v = g.ndata['nx_id'][etype[2]][v]
        new_etype = (new_node_type, str(rel2idx[etype[1]]), new_node_type)
        edge_dict[new_etype] = (new_u, new_v)
    edge_dict[('no_type','0','no_type')]=(torch.tensor([g.num_nodes()]),torch.tensor([g.num_nodes()]))
    
    for exp in pred.items():
        u, v = exp[0][0], exp[0][1]
        new_u = g.ndata['nx_id'][u[0]][u[1]]
        new_v = g.ndata['nx_id'][v[0]][v[1]]
        trace_u = torch.empty(0, dtype=torch.int64)
        trace_v = torch.empty(0, dtype=torch.int64)
        trace_r = torch.empty(0, dtype=torch.int64)
        for edge in exp[1].items():
            edge_u = edge[1][0]
            edge_v = edge[1][1]
            new_edge_u = g.ndata['nx_id'][edge[0][0]][edge_u]
            new_edge_v = g.ndata['nx_id'][edge[0][2]][edge_v]
            r = torch.tensor([rel2idx[edge[0][1]]]*new_edge_u.shape[0])
            trace_u = torch.cat([trace_u, new_edge_u])
            trace_v = torch.cat([trace_v, new_edge_v])
            trace_r = torch.cat([trace_r, r])

        pred_dict[(new_u.item(), new_v.item())] = torch.cat([trace_u.unsqueeze(0), trace_r.unsqueeze(0), trace_v.unsqueeze(0)], dim=0)

    pred_path_dict = {}
    for exp in pred_path.items():
        u, v = exp[0][0], exp[0][1]
        new_u = g.ndata['nx_id'][u[0]][u[1]]
        new_v = g.ndata['nx_id'][v[0]][v[1]]
        trace_u = torch.empty(0, dtype=torch.int64)
        trace_v = torch.empty(0, dtype=torch.int64)
        trace_r = torch.empty(0, dtype=torch.int64)
        i=0
        for path in exp[1]:
            for edge in path:
                edge_u = edge[1]
                edge_v = edge[2]
                new_edge_u = g.ndata['nx_id'][edge[0][0]][edge_u].unsqueeze(0)
                new_edge_v = g.ndata['nx_id'][edge[0][2]][edge_v].unsqueeze(0)
                r = torch.tensor([i])
                trace_u = torch.cat([trace_u, new_edge_u])
                trace_v = torch.cat([trace_v, new_edge_v])
                trace_r = torch.cat([trace_r, r])
            i+=1

        pred_path_dict[(new_u.item(), new_v.item())] = torch.cat([trace_u.unsqueeze(0), trace_r.unsqueeze(0), trace_v.unsqueeze(0)], dim=0)

    new_graph = dgl.heterograph(edge_dict, num_nodes_dict={new_node_type: g.num_nodes()+2})
    return new_graph, pred_dict, rel2idx, pred_path_dict

def load_dataset_pg(dataset_dir, dataset_name, val_ratio, test_ratio):    
    graph_saving_path = f'{dataset_dir}/{dataset_name}'
    graph_list, _ = dgl.load_graphs(graph_saving_path)
    pred_pair_to_edge_labels = torch.load(f'{graph_saving_path}_pred_pair_to_edge_labels')
    pred_pair_to_path_labels = torch.load(f'{graph_saving_path}_pred_pair_to_path_labels')
    g = graph_list[0]

    i=0
    node_dict = {}
    for ntype in g.ntypes:
        node_dict[ntype]=torch.tensor(range(i,i+g.num_nodes(ntype)))
        i+=g.num_nodes(ntype)
    g.ndata['nx_id']=node_dict

    new_g, pred, rel2idx, pred_path = get_nodehomo_graph(g, pred_pair_to_edge_labels, pred_pair_to_path_labels)
    torch.save(pred, f'{graph_saving_path}_pred_pair_to_edge_labels_homo')
    torch.save(pred_path, f'{graph_saving_path}_pred_pair_to_path_labels_homo')

    pred_etype = str(rel2idx['likes'])
    neg_etype = str(0)

    process_data(new_g, val_ratio, test_ratio, pred_etype, graph_saving_path)
    print(rel2idx)
    
    eids = new_g.edges('eid', etype=pred_etype)
    removed_ghetero = dgl.remove_edges(new_g, eids, etype=pred_etype)
    eids = removed_ghetero.edges('eid', etype=neg_etype)
    removed_ghetero = dgl.remove_edges(removed_ghetero, eids, etype=neg_etype)
    dgl.save_graphs(f'{graph_saving_path}_homo', removed_ghetero)
    return

def load_dataset_pg1(dataset_dir, dataset_name):
    graph_saving_path = f'{dataset_dir}/{dataset_name}'
    graph_list, _ = dgl.load_graphs(f'{graph_saving_path}_homo')
    pred_pair_to_edge_labels = torch.load(f'{graph_saving_path}_pred_pair_to_edge_labels_homo')
    pred_pair_to_path_labels = torch.load(f'{graph_saving_path}_pred_pair_to_path_labels_homo')
    
    train_link = torch.load(f'{graph_saving_path}_train_link')
    test_link = torch.load(f'{graph_saving_path}_test_link')
    g = graph_list[0]


    return g, train_link, test_link, pred_pair_to_edge_labels, pred_pair_to_path_labels

