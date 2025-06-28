#!/usr/bin/env python3
import dgl
import torch
import random
import textwrap
import yaml
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
from dgl.subgraph import khop_in_subgraph
from itertools import count
from heapq import heappop, heappush
from sklearn.metrics import roc_auc_score

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def print_args(args):
    for k, v in vars(args).items():
        print(f'{k:25} {v}')
        
def set_config_args(args, config_path, dataset_name, model_name=''):
    with open(config_path, "r") as conf:
        config = yaml.load(conf, Loader=yaml.FullLoader)[dataset_name]
        if model_name:
            config = config[model_name]

    for key, value in config.items():
        setattr(args, key, value)
    return args
    
'''
Model training utils
'''
def idx_split(idx, ratio, seed=0):
    """
    Randomly split `idx` into idx1 and idx2, where idx1 : idx2 = `ratio` : 1 - `ratio`
    
    Parameters
    ----------
    idx : tensor
        
    ratio: float
 
    Returns
    ----------
        Two index (tensor) after split
    """
    set_seed(seed)
    n = len(idx)
    cut = int(n * ratio)
    idx_idx_shuffle = torch.randperm(n)

    idx1_idx, idx2_idx = idx_idx_shuffle[:cut], idx_idx_shuffle[cut:]
    idx1, idx2 = idx[idx1_idx], idx[idx2_idx]
    assert((torch.cat([idx1, idx2]).sort()[0] == idx.sort()[0]).all())
    return idx1, idx2


def eids_split(eids, val_ratio, test_ratio, seed=0):
    """
    Split `eids` into three parts: train, valid, and test,
    where train : valid : test = (1 - `val_ratio` - `test_ratio`) : `val_ratio` : `test_ratio`
    
    Parameters
    ----------
    eid : tensor
        edge id
        
    val_ratio : float
    
    test_ratio : float

    seed : int

    Returns
    ----------
        Three edge ids (tensor) after split
    """
    train_ratio = (1 - val_ratio - test_ratio)
    train_eids, pred_eids = idx_split(eids, train_ratio, seed)
    val_eids, test_eids = idx_split(pred_eids, val_ratio / (1 - train_ratio), seed)
    return train_eids, val_eids, test_eids

def negative_sampling(graph, pred_etype=None, num_neg_samples=None):
    # src_N: total number of src nodes
    # N (tgt_N): total number of tgt nodes
    # M: total number of possible edges, square of src_N * tgt_N
    # pos_M: number of positive samples (observed edges)
    # neg_M: number of negative samples
    pos_src_nids, pos_tgt_nids = graph.edges(etype=pred_etype)
    if pred_etype is None:
        N = graph.num_nodes()
        M = N * N
    else:
        src_ntype, _, tgt_ntype = graph.to_canonical_etype(pred_etype) 
        src_N, N = graph.num_nodes(src_ntype), graph.num_nodes(tgt_ntype)
        M = src_N * N

    pos_M = pos_src_nids.shape[0]
    neg_M = num_neg_samples or pos_M
    neg_M = min(neg_M, M - pos_M) # incase M - pos_M < neg_M

    # Percentage of edges to opos_tgt_nidsersample, so only need to sample once in most cases
    alpha = abs(1 / (1 - 1.1 * (pos_M / M)))
    size = min(M, int(alpha * neg_M))
    perm = torch.tensor(random.sample(range(M), size))
    
    idx = pos_src_nids * N + pos_tgt_nids
    # mask = torch.from_npos_src_nidsmpy(np.isin(perm, idx.to('cppos_src_nids'))).to(torch.bool)
    mask = torch.isin(perm, idx.to('cpu')).to(torch.bool)
    perm = perm[~mask][:neg_M].to(pos_src_nids.device)

    neg_src_nids = torch.div(perm, N, rounding_mode='floor')
    neg_tgt_nids = perm % N

    return neg_src_nids, neg_tgt_nids

'''
DGL graph manipulation utils
'''
def get_homo_nids_to_hetero_nids(ghetero):
    '''
    Create a dictionary mapping the node ids of the homogeneous version of the input graph
    to the node ids of the input heterogeneous graph.
    
    Parameters
    ----------
    ghetero : heterogeneous dgl graph
        
    Returns
    ----------
    homo_nids_to_hetero_nids : dict
    '''
    ghomo = dgl.to_homogeneous(ghetero)
    homo_nids = range(ghomo.num_nodes())
    hetero_nids = ghomo.ndata[dgl.NID].tolist()
    homo_nids_to_hetero_nids = dict(zip(homo_nids, hetero_nids))
    return homo_nids_to_hetero_nids

def get_homo_nids_to_ntype_hetero_nids(ghetero):
    '''
    Create a dictionary mapping the node ids of the homogeneous version of the input graph
    to tuples as (node type, node id) of the input heterogeneous graph.
    
    Parameters
    ----------
    ghetero : heterogeneous dgl graph
        
    Returns
    ----------
    homo_nids_to_ntype_hetero_nids : dict
    '''
    ghomo = dgl.to_homogeneous(ghetero)
    homo_nids = range(ghomo.num_nodes())
    ntypes = ghetero.ntypes
    # This line relies on the default order of ntype_ids is the order in ghetero.ntypes
    ntypes = [ntypes[i] for i in ghomo.ndata[dgl.NTYPE]] 
    hetero_nids = ghomo.ndata[dgl.NID].tolist()
    ntypes_hetero_nids = list(zip(ntypes, hetero_nids))
    homo_nids_to_ntype_hetero_nids = dict(zip(homo_nids, ntypes_hetero_nids))
    return homo_nids_to_ntype_hetero_nids

def get_ntype_hetero_nids_to_homo_nids(ghetero):
    '''
    Create a dictionary mapping tuples as (node type, node id) of the input heterogeneous graph
    to the node ids of the homogeneous version of the input graph.
    
    Parameters
    ----------
    ghetero : heterogeneous dgl graph
        
    Returns
    ----------
    ntype_hetero_nids_to_homo_nids : dict
    '''
    tmp = get_homo_nids_to_ntype_hetero_nids(ghetero)
    ntype_hetero_nids_to_homo_nids = {v: k for k, v in tmp.items()}
    return ntype_hetero_nids_to_homo_nids

def get_ntype_pairs_to_cannonical_etypes(ghetero, pred_etype='likes'):
    '''
    Create a dictionary mapping tuples as (source node type, target node type) to 
    cannonical edge types. Edges wity type `pred_etype` will be excluded.
    A helper function for path finding.
    Only works if there is only one edge type between any pair of node types.
    
    Parameters
    ----------
    ghetero : heterogeneous dgl graph
      
    pred_etype : string
        The edge type for prediction

    Returns
    ----------
    ntype_pairs_to_cannonical_etypes : dict
    '''
    ntype_pairs_to_cannonical_etypes = {}
    for src_ntype, etype, tgt_ntype in ghetero.canonical_etypes:
        if etype != pred_etype:
            ntype_pairs_to_cannonical_etypes[(src_ntype, tgt_ntype)] = (src_ntype, etype, tgt_ntype)
    return ntype_pairs_to_cannonical_etypes

def get_num_nodes_dict(ghetero):
    '''
    Create a dictionary containing number of nodes of all ntypes in a heterogeneous graph
    Parameters
    ----------
    ghetero : heterogeneous dgl graph

    Returns 
    ----------
    num_nodes_dict : dict
        key=node type, value=number of nodes
    '''
    num_nodes_dict = {}
    for ntype in ghetero.ntypes:
        num_nodes_dict[ntype] = ghetero.num_nodes(ntype)    
    return num_nodes_dict

def remove_all_edges_of_etype(ghetero, etype):
    '''
    Remove all edges with type `etype` from `ghetero`. If `etype` is not in `ghetero`, do nothing.
    
    Parameters
    ----------
    ghetero : heterogeneous dgl graph

    etype : string or triple of strings
        Edge type in simple form (string) or cannonical form (triple of strings)
    
    Returns 
    ----------
    removed_ghetero : heterogeneous dgl graph
        
    '''
    etype = ghetero.to_canonical_etype(etype)
    if etype in ghetero.canonical_etypes:
        eids = ghetero.edges('eid', etype=etype)
        removed_ghetero = dgl.remove_edges(ghetero, eids, etype=etype)
    else:
        removed_ghetero = ghetero
    return removed_ghetero

def hetero_src_tgt_khop_in_subgraph(src_nid, tgt_nid, ghetero, k):
    # Extract k-hop subgraph centered at the (src, tgt) pair
    src_nid = src_nid.item() if torch.is_tensor(src_nid) else src_nid
    tgt_nid = tgt_nid.item() if torch.is_tensor(tgt_nid) else tgt_nid
    

    pred_dict = {'no_type': torch.tensor([src_nid, tgt_nid])}
    sghetero, inv_dict = khop_in_subgraph(ghetero, pred_dict, k)
    sghetero_src_nid = inv_dict['no_type'][0]
    sghetero_tgt_nid = inv_dict['no_type'][1]

    sghetero_feat_nid = sghetero.ndata[dgl.NID]
    
    return sghetero_src_nid, sghetero_tgt_nid, sghetero, sghetero_feat_nid


'''
Path finding utils
'''
def get_neg_path_score_func(g, weight, exclude_node=[]):
    '''
    Compute the negative path score for the shortest path algorithm.
    
    Parameters
    ----------
    g : dgl graph

    weight: string
       The edge weights stored in g.edata

    exclude_node : iterable
        Degree of these nodes will be set to 0 when computing the path score, so they will likely be included.

    Returns
    ----------
    neg_path_score_func: callable function
       Takes in two node ids and return the edge weight. 
    '''
    log_eweights = g.edata[weight].log().tolist()
    log_in_degrees = g.in_degrees().log()
    log_in_degrees[exclude_node] = 0
    log_in_degrees = log_in_degrees.tolist()
    u, v = g.edges()
    neg_path_score_map = {edge : log_in_degrees[edge[1]] - log_eweights[i] for i, edge in enumerate(zip(u.tolist(), v.tolist()))}

    def neg_path_score_func(u, v):
        return neg_path_score_map[(u, v)]
    return neg_path_score_func

def bidirectional_dijkstra(g, src_nid, tgt_nid, weight=None, ignore_nodes=None, ignore_edges=None):
    if src_nid == tgt_nid:
        return (0, [src_nid])

    src, tgt = g.edges()
    Gpred = lambda i: src[tgt == i].tolist()
    Gsucc = lambda i: tgt[src == i].tolist()
    
    if ignore_nodes:
        def filter_iter(nodes):
            def iterate(v):
                for w in nodes(v):
                    if w not in ignore_nodes:
                        yield w

            return iterate

        Gpred = filter_iter(Gpred)
        Gsucc = filter_iter(Gsucc)
    
    if ignore_edges:
        def filter_pred_iter(pred_iter):
            def iterate(v):
                for w in pred_iter(v):
                    if (w, v) not in ignore_edges:
                        yield w

            return iterate

        def filter_succ_iter(succ_iter):
            def iterate(v):
                for w in succ_iter(v):
                    if (v, w) not in ignore_edges:
                        yield w

            return iterate

        Gpred = filter_pred_iter(Gpred)
        Gsucc = filter_succ_iter(Gsucc)

    push = heappush
    pop = heappop
    # Init:   Forward             Backward
    dists = [{}, {}]  # dictionary of final distances
    paths = [{src_nid: [src_nid]}, {tgt_nid: [tgt_nid]}]  # dictionary of paths
    fringe = [[], []]  # heap of (distance, node) tuples for
    # extracting next node to expand
    seen = [{src_nid: 0}, {tgt_nid: 0}]  # dictionary of distances to
    # nodes seen
    c = count()
    # initialize fringe heap
    push(fringe[0], (0, next(c), src_nid))
    push(fringe[1], (0, next(c), tgt_nid))
    # neighs for extracting correct neighbor information
    neighs = [Gsucc, Gpred]
    # variables to hold shortest discovered path
    # finaldist = 1e30000
    finalpath = []
    dir = 1
    if not weight:
        weight = lambda u, v: 1
            
    while fringe[0] and fringe[1]:
        # choose direction
        # dir == 0 is forward direction and dir == 1 is back
        dir = 1 - dir
        # extract closest to expand
        (dist, _, v) = pop(fringe[dir])
        if v in dists[dir]:
            # Shortest path to v has already been found
            continue
        # update distance
        dists[dir][v] = dist  # equal to seen[dir][v]
        if v in dists[1 - dir]:
            # if we have scanned v in both directions we are done
            # we have now discovered the shortest path
            return (finaldist, finalpath)

        for w in neighs[dir](v):
            if dir == 0:  # forward
                minweight = weight(v, w)
                vwLength = dists[dir][v] + minweight
            else:  # back, must remember to change v,w->w,v
                minweight = weight(w, v)
                vwLength = dists[dir][v] + minweight

            if w in dists[dir]:
                if vwLength < dists[dir][w]:
                    raise ValueError("Contradictory paths found: negative weights?")
            elif w not in seen[dir] or vwLength < seen[dir][w]:
                # relaxing
                seen[dir][w] = vwLength
                push(fringe[dir], (vwLength, next(c), w))
                paths[dir][w] = paths[dir][v] + [w]
                if w in seen[0] and w in seen[1]:
                    # see if this path is better than the already
                    # discovered shortest path
                    totaldist = seen[0][w] + seen[1][w]
                    if finalpath == [] or finaldist > totaldist:
                        finaldist = totaldist
                        revpath = paths[1][w][:]
                        revpath.reverse()
                        finalpath = paths[0][w] + revpath[1:]
    raise ValueError("No paths found")


class PathBuffer:
    """For shortest paths finding
    
    Adapted from NetworkX shortest_simple_paths
    https://networkx.org/documentation/stable/_modules/networkx/algorithms/simple_paths.html

    """
    def __init__(self):
        self.paths = set()
        self.sortedpaths = list()
        self.counter = count()

    def __len__(self):
        return len(self.sortedpaths)

    def push(self, cost, path):
        hashable_path = tuple(path)
        if hashable_path not in self.paths:
            heappush(self.sortedpaths, (cost, next(self.counter), path))
            self.paths.add(hashable_path)

    def pop(self):
        (cost, num, path) = heappop(self.sortedpaths)
        hashable_path = tuple(path)
        self.paths.remove(hashable_path)
        return path
    
def k_shortest_paths_generator(g, 
                               src_nid, 
                               tgt_nid, 
                               weight=None, 
                               k=5, 
                               ignore_nodes_init=None,
                               ignore_edges_init=None):
    if not weight:
        weight = lambda u, v: 1

    def length_func(path):
        return sum(weight(u, v) for (u, v) in zip(path, path[1:]))

    listA = list()
    listB = PathBuffer()
    prev_path = None
    while not prev_path or len(listA) < k:
        if not prev_path:
            length, path = bidirectional_dijkstra(g, src_nid, tgt_nid, weight, ignore_nodes_init, ignore_edges_init)
            listB.push(length, path)
        else:
            ignore_nodes = set(ignore_nodes_init) if ignore_nodes_init else set()
            ignore_edges = set(ignore_edges_init) if ignore_edges_init else set()
            for i in range(1, len(prev_path)):
                root = prev_path[:i]
                root_length = length_func(root)
                for path in listA:
                    if path[:i] == root:
                        ignore_edges.add((path[i - 1], path[i]))
                try:
                    length, spur = bidirectional_dijkstra(g,
                                                          root[-1],
                                                          tgt_nid,
                                                          ignore_nodes=ignore_nodes,
                                                          ignore_edges=ignore_edges,
                                                          weight=weight)
                    path = root[:-1] + spur
                    listB.push(root_length + length, path)
                except ValueError:
                    pass
                ignore_nodes.add(root[-1])
        
        if listB:
            path = listB.pop()
            yield path
            listA.append(path)
            prev_path = path
        else:
            break

def k_shortest_paths_with_max_length(g, 
                                     src_nid, 
                                     tgt_nid, 
                                     weight=None, 
                                     k=5, 
                                     max_length=None,
                                     ignore_nodes=None,
                                     ignore_edges=None):
    
    """Generate at most `k` simple paths in the graph g from src_nid to tgt_nid,
       each with maximum lenghth `max_length`, return starting from the shortest ones. 
       If a weighted shortest path search is to be used, no negative weights are allowed.
   
    Parameters
    ----------
       See function `k_shortest_paths_generator`
   
    Return
    -------
    paths: list of lists
       Each list is a path containing node ids
    """
    path_generator = k_shortest_paths_generator(g, 
                                                src_nid, 
                                                tgt_nid, 
                                                weight=weight,
                                                k=k, 
                                                ignore_nodes_init=ignore_nodes,
                                                ignore_edges_init=ignore_edges)
    
    try:
        if max_length:
            paths = [path for path in path_generator if len(path) <= max_length + 1]
        else:
            paths = list(path_generator)

    except ValueError:
        paths = [[]]

    return paths

'''
Evaluation utils
'''
def get_comp_g_edge_labels(comp_g, edge_labels):
    """Turn `edge_labels` with node ids in the original graph to
       `comp_g_edge_labels` with node ids in the computation graph.
       For easier evaluation.

    Parameters
    ----------
    comp_g : heterogeneous dgl graph
        computation graph, with .ndata stores key dgl.NID
    
    edge_labels : dict
        key=edge type, value=(source node ids, target node ids)
   
    Return
    -------
    comp_g_edge_labels: dict
        key=edge type, value=a tensor of labels, each label is in {0, 1}
    """
    ntype_to_tensor_nids_to_comp_g_nids = {}
    ntypes_to_comp_g_max_nids = {}
    ntypes_to_nids = comp_g.ndata[dgl.NID]
    for ntype in ntypes_to_nids.keys():
        nids = ntypes_to_nids[ntype]
        if nids.numel() > 0:
            max_nid = nids.max().item()
        else: 
            max_nid = -1

        ntypes_to_comp_g_max_nids[ntype] = max_nid

        nids_to_comp_g_nids = torch.zeros(max_nid + 1).long() - 1
        # The i-th entry will be the nid in comp_g for the i-th node in g
        nids_to_comp_g_nids[nids] = torch.arange(nids.shape[0])
        ntype_to_tensor_nids_to_comp_g_nids[ntype] = nids_to_comp_g_nids


    comp_g_edge_labels = {}
    for can_etype in edge_labels:
        start_ntype, etype, end_ntype = can_etype
        start_nids, end_nids = edge_labels[can_etype]
        start_comp_g_max_nid, end_comp_g_max_nid = ntypes_to_comp_g_max_nids[start_ntype], ntypes_to_comp_g_max_nids[end_ntype]

        # For edges in label but not in comp_g, exclude them
        start_included_nid_mask = start_nids <= start_comp_g_max_nid
        end_included_nid_mask = end_nids <= end_comp_g_max_nid
        comp_g_included_nid_mask = end_included_nid_mask & start_included_nid_mask

        start_nids = start_nids[comp_g_included_nid_mask]
        end_nids = end_nids[comp_g_included_nid_mask]

        comp_g_start_nids = ntype_to_tensor_nids_to_comp_g_nids[start_ntype][start_nids]
        comp_g_end_nids = ntype_to_tensor_nids_to_comp_g_nids[end_ntype][end_nids]
        comp_g_eids = comp_g.edge_ids(comp_g_start_nids.tolist(), comp_g_end_nids.tolist(), etype=etype)

        num_edges = comp_g.num_edges(etype=can_etype)
        comp_g_eid_mask = torch.zeros(num_edges)
        comp_g_eid_mask[comp_g_eids] = 1

        comp_g_edge_labels[can_etype] = comp_g_eid_mask

    return comp_g_edge_labels    

def get_comp_g_path_labels(comp_g, path_labels):
    """Turn `path_labels` with node ids in the original graph
       `comp_g_path_labels` with node ids in the computation graph
       For easier evaluation.

    Parameters
    ----------
    comp_g : heterogeneous dgl graph
        computation graph, with .ndata stores key dgl.NID
    
    path_labels : list of lists
        Each list is a path, i.e., triples of 
        (cannonical edge type, source node id, target node id)
   
    Returns
    -------
    comp_g_path_labels: list of lists
        Each list is a path, i.e., tuples of (cannonical edge type, edge id)
    """
    ntype_to_tensor_nids_to_comp_g_nids = {}
    ntypes_to_comp_g_max_nids = {}
    ntypes_to_nids = comp_g.ndata[dgl.NID]
    for ntype in ntypes_to_nids.keys():
        nids = ntypes_to_nids[ntype]
        if nids.numel() > 0:
            max_nid = nids.max().item()
        else: 
            max_nid = -1

        ntypes_to_comp_g_max_nids[ntype] = max_nid

        nids_to_comp_g_nids = torch.zeros(max_nid + 1).long() - 1
        # The i-th entry will be the nid in comp_g for the i-th node in g
        nids_to_comp_g_nids[nids] = torch.arange(nids.shape[0])
        ntype_to_tensor_nids_to_comp_g_nids[ntype] = nids_to_comp_g_nids

    comp_g_path_labels = []
    for path in path_labels:
        comp_g_path = []
        for can_etype, start_nid, end_nid in path:
            start_ntype, etype, end_ntype = can_etype

            comp_g_start_nid = ntype_to_tensor_nids_to_comp_g_nids[start_ntype][start_nid].item()
            comp_g_end_nid = ntype_to_tensor_nids_to_comp_g_nids[end_ntype][end_nid].item()

            comp_g_eid = comp_g.edge_ids(comp_g_start_nid, comp_g_end_nid, etype=can_etype)
            comp_g_path += [(can_etype, comp_g_eid)]
        comp_g_path_labels += [comp_g_path]
    return comp_g_path_labels

def eval_edge_mask_auc(edge_mask_dict, edge_labels):
    '''
    Evaluate the AUC of an edge mask
    
    Parameters
    ----------
    edge_mask_dict: dict
        key=edge type, value=a tensor of labels, each label is in (-inf, inf)

    edge_labels: dict
        key=edge type, value=a tensor of labels, each label is in {0, 1}

    Returns
    ----------
    ROC-AUC score : int
    '''
    
    y_true = []
    y_score = []
    for can_etype in edge_labels:
        y_true += [edge_labels[can_etype]]
        y_score += [edge_mask_dict[can_etype].detach().sigmoid()]

    y_true = torch.cat(y_true)
    y_score = torch.cat(y_score)
    
    return roc_auc_score(y_true, y_score) 

def eval_edge_mask_topk_path_hit(edge_mask_dict, path_labels, topks=[10]):
    '''
    Evaluate the path hit rate of the top k edges in an edge mask
    
    Parameters
    ----------
    edge_mask_dict: dict
        key=edge type, value=a tensor of labels, each label is in (-inf, inf)

    path_labels: list of lists
        Each list is a path, i.e., tuples of (cannonical edge type, edge id)

    topks: iterable
        An iterable of the top `k` values. Each `k` determines how many edges to select 
        from the top values of the mask.

    Returns
    ----------
    topk_to_path_hit: dict
        Mapping the top `k` to 
    '''
    cat_edge_mask = torch.cat([v for v in edge_mask_dict.values()])
    M = len(cat_edge_mask)
    topks = {k: min(k, M) for k in topks}

    topk_to_path_hit = defaultdict(list)
    for r, k in topks.items():
        threshold = cat_edge_mask.topk(k)[0][-1].item()
        hard_edge_mask_dict = {}
        for etype in edge_mask_dict:
            hard_edge_mask_dict[etype] = edge_mask_dict[etype] >= threshold

        hit = eval_hard_edge_mask_path_hit(hard_edge_mask_dict, path_labels)
        topk_to_path_hit[r] += [hit]
    return topk_to_path_hit

def eval_hard_edge_mask_path_hit(hard_edge_mask_dict, path_labels):
    '''
    Evaluate the path hit of the an hard edge mask
    
    Parameters
    ----------
    hard_edge_mask_dict: dict
        key=edge type, value=a tensor of labels, each label is in {True, False}

    path_labels: list of lists
        Each list is a path, i.e., tuples of (cannonical edge type, edge id)

    Returns
    ----------
    hit_path: int
        1 or 0
    '''
    for path in path_labels:
        hit_path = 1
        for can_etype, eid in path:
            if not hard_edge_mask_dict[can_etype][eid]:
                hit_path = 0
                break
        if hit_path:
            return 1
    return 0


def eval_path_explanation_edges_path_hit(path_explanation_edges, path_labels):
    '''
    Evaluate the path hit rate of the a path_explanation_edges
    
    Parameters
    ----------
    path_explanation_edges : list
        Edges on the path explanation, each edge is a triples of 
        (cannonical edge type, source node id, target node id)
    
    path_labels : list of lists
        Each list is a path, i.e., triples of 
        (cannonical edge type, source node id, target node id)

    Returns
    ----------
    hit_path: int
        1 or 0
    '''
    for path in path_labels:
        hit_path = 1
        for edge in path:
            if edge not in path_explanation_edges:
                hit_path = 0
                break
        if hit_path:
            return 1
    return 0


'''
Plotting utils
'''
def plot_hetero_graph(ghetero,
                      ntypes_to_nshapes=None,
                      ntypes_to_ncolors=None,
                      ntypes_to_nlayers=None,
                      layout='multipartite',
                      layout_seed=0,
                      node_size=1000,
                      edge_kwargs={},
                      selected_node_dict=None,
                      selected_node_color='red',
                      selected_edge_dict=None,
                      selected_edge_kwargs={},
                      label='nid',
                      etype_label=True,
                      label_offset=False,
                      title=None,
                      legend=False,
                      figsize=(10, 10),
                      fig_name=None,
                      fig_format='png',
                      is_show=True):
        '''
        Parameters
        ----------
        ghetero: a DGL heterogeneous graph with ndata `order`

        ntypes_to_nshapes : Dict
            mapping node types to node shapes
        
        ntypes_to_ncolors : Dict
            mapping node types to node colors

        ntypes_to_nlayers : Dict 
            mapping node types to layer order in the multipartite layout. 

        label: String
            one of ['none', nid'] or a node feature stored in ndata of ghetero

        Returns
        ----------
        nx_graph : networkx graph
        
        '''
        if ntypes_to_nshapes is None:
            default_node_shape = 'o'
        if ntypes_to_ncolors is None:
            default_node_color = 'cyan'
        if selected_node_dict is not None:
            selected_node_dict = {ntype: list(selected_node_dict[ntype]) for ntype in selected_node_dict}

        # Convert DGL graph to networkx graph
        ghomo = dgl.to_homogeneous(ghetero)
        edges = torch.cat([t.unsqueeze(1) for t in ghomo.edges()], dim=1)
        edge_list = [(n_frm, n_to) for (n_frm, n_to) in edges.tolist()]
        nx_graph = dgl.to_networkx(ghomo, node_attrs=[dgl.NTYPE])
            
        # Use different layout
        if layout == 'spring':
            pos = nx.spring_layout(nx_graph, seed=layout_seed)
        elif layout == 'kk':
            pos = nx.kamada_kawai_layout(nx_graph)
        elif layout == 'multipartite':
            if ntypes_to_nlayers is not None:
                ntype_ids_to_nlayers = {ghetero.get_ntype_id(ntype): ntypes_to_nlayers[ntype] for ntype in ghetero.ntypes}
            else:
                ntype_ids_to_nlayers = {ghetero.get_ntype_id(ntype): i for i, ntype in enumerate(ghetero.ntypes)}
                
            for i in nx_graph.nodes():
                ntype_id = nx_graph.nodes()[i][dgl.NTYPE].item()
                nx_graph.nodes()[i][dgl.NTYPE] = ntype_ids_to_nlayers[ntype_id]

            pos = nx.multipartite_layout(nx_graph, subset_key=dgl.ETYPE, scale=1)
        else:
            raise ValueError('Unknown layout')

        # Start drawing
        plt.figure(figsize=figsize)
        ax = plt.gca()
 
        # Draw nodes for each ntype
        for ntype in ghetero.ntypes:
            ntype_ids = ghomo.ndata[dgl.NTYPE]
            hetero_nids = ghomo.ndata[dgl.NID] # nid in the original hetero graph
            
            node_shape = ntypes_to_nshapes[ntype] if ntypes_to_nshapes else default_node_shape
            node_color = ntypes_to_ncolors[ntype] if ntypes_to_ncolors else default_node_color

            # For the current node type, get the node type id and node ids
            curr_ntype_id = ghetero.get_ntype_id(ntype)
            curr_nids_mask = ntype_ids == curr_ntype_id
            curr_nids = curr_nids_mask.nonzero().view(-1).tolist()

            # For the current node type, get node ids and prediction node id in the original hetero graph
            curr_hetero_nids = hetero_nids[curr_nids_mask]
            
            if selected_node_dict is not None:
                curr_hetero_selected_nid = selected_node_dict.get(ntype)
                if curr_hetero_selected_nid is not None:
                    curr_node_color = []
                    for hetero_nid in curr_hetero_nids:
                        curr_node_color += [selected_node_color if hetero_nid in curr_hetero_selected_nid else node_color]
                    node_color = curr_node_color

            nx.draw_networkx_nodes(nx_graph, 
                                   pos, 
                                   curr_nids, 
                                   node_shape=node_shape,
                                   node_color=node_color,
                                   node_size=node_size,
                                   ax=ax)
            
        # Draw edges
        nx.draw_networkx_edges(nx_graph, pos, edge_list, **edge_kwargs, ax=ax)
        
        if selected_edge_dict is not None:
            ntype_hetero_nids_to_homo_nids = get_ntype_hetero_nids_to_homo_nids(ghetero)
            homo_selected_edge_list = []
            for etype in selected_edge_dict:
                src_ntype, _, tgt_ntype = ghetero.to_canonical_etype(etype)
                src_nids, tgt_nids = selected_edge_dict[etype]
                for src_nid, tgt_nid in zip(src_nids.tolist(), tgt_nids.tolist()):
                    homo_src_nid = ntype_hetero_nids_to_homo_nids[(src_ntype, src_nid)]
                    homo_tgt_nid = ntype_hetero_nids_to_homo_nids[(tgt_ntype, tgt_nid)]
                    homo_selected_edge_list += [(homo_src_nid, homo_tgt_nid)]
        
            nx.draw_networkx_edges(nx_graph, pos, homo_selected_edge_list, **selected_edge_kwargs, ax=ax)
            
            
     # Start labelling nodes
        if label == 'none':
            pass
        elif label == 'nid':
            homo_nids_to_hetero_nids = get_homo_nids_to_hetero_nids(ghetero)
            nx.draw_networkx_labels(nx_graph, pos, labels=homo_nids_to_hetero_nids)
        else:
            # Set extra space to avoid label outside of the box
            x_values, y_values = zip(*pos.values())
            x_max = max(x_values)
            x_min = min(x_values)
            x_margin = (x_max - x_min) * 0.12
            ax.set_xlim(x_min - x_margin, x_max + x_margin)


            if ghetero.ndata.get(label):
                homo_nids_to_hetero_ndata_feat = get_homo_nids_to_hetero_ntype_data_feat(ghetero, label)
                if label_offset:
                    offset = 0.8 / figsize[1]
                    label_pos = {nid : [p[0], p[1] - offset] for nid, p in pos.items()} 
                else:
                    label_pos = pos

                nx.draw_networkx_labels(nx_graph, 
                                        label_pos, 
                                        font_size=14, 
                                        font_weight='bold', 
                                        labels=homo_nids_to_hetero_ndata_feat,
                                        horizontalalignment='center',
                                        verticalalignment='center',
                                        ax=ax)

            else:
                raise ValueError('Unrecognized label')
            
        # Start labelling edges with etype
        if etype_label is not None:
            if ghetero.ndata.get(label):
                homo_nid_pairs_to_etypes = get_homo_nid_pairs_to_etypes(ghetero)
                nx.draw_networkx_edge_labels(nx_graph, 
                                             pos, 
                                             font_size=13, 
                                             font_weight='bold', 
                                             edge_labels=homo_nid_pairs_to_etypes,
                                             horizontalalignment='center',
                                             verticalalignment='center',
                                             ax=ax)
            
        if legend:
            plt.legend(ghetero.ntypes, fontsize=15, prop={'size': figsize[0]*2.5}, bbox_to_anchor = (1.15, 0.7)) 

        ax.axis('off')
        if title is not None:
            plt.title(textwrap.fill(title, width=60))
        if fig_name is not None:
            plt.savefig(fig_name, format=fig_format, bbox_inches='tight')
        if is_show:
            plt.show()
        if fig_name is not None:
            plt.close()
            
        return nx_graph
  
def get_homo_nids_to_hetero_ntype_data_feat(ghetero, feat=dgl.NID):
    '''
    Plotting helper function
    '''
    ghomo = dgl.to_homogeneous(ghetero)
    homo_nids = range(ghomo.num_nodes())
    hetero_ndata_feat = []
    for ntype in ghetero.ntypes:
        hetero_ndata_feat += [f'{ntype[0]}' + f'{feat}' for feat in ghetero.ndata[feat][ntype].tolist()]

    homo_nids_to_hetero_ndata_feat = dict(zip(homo_nids, hetero_ndata_feat))
    return homo_nids_to_hetero_ndata_feat

def get_homo_nid_pairs_to_etypes(ghetero):
    '''
    Plotting helper function
    '''
    ghomo = dgl.to_homogeneous(ghetero)
    etypes = ghetero.etypes
    etype_list = [etypes[etype_id] for etype_id in ghomo.edata[dgl.ETYPE]]
    u, v = ghomo.edges()
    homo_nid_pairs_to_etypes = dict(zip(zip(u.tolist(), v.tolist()), etype_list))
    return homo_nid_pairs_to_etypes




import numpy as np
import torch

def get_longest_trace(data, rule):

    if rule == 'full_data':
        longest_trace = data['max_trace']
    elif rule == 'test_full':
        longest_trace = data['max_trace']
    else:
        longest_trace = data[rule + '_longest_trace']

    return longest_trace

def graded_precision_recall(
    true_exp,
    pred_exp,
    true_weight,
    unk_ent_id,
    unk_rel_id,
    unk_weight_id):

    '''
    pred_exp: numpy array without padding
    true_exp: numpy array with padding

    '''
    
    n = len(pred_exp)
    
    unk = np.array([[unk_ent_id, unk_rel_id, unk_ent_id]])

    #first compute number of triples in explanation (must exclude padded triples)
    num_explanations = 0

    #number of triples per explanation
    num_true_triples = []

    for i in range(len(true_exp)):

        current_trace = true_exp[i]

        num_triples = (current_trace != unk).all(axis=1).sum()

        if  num_triples > 0:

            num_explanations += 1
            num_true_triples.append(num_triples)

    num_true_triples = np.array(num_true_triples)

    relevance_scores = np.zeros(num_explanations)

    for i in range(n):

        current_pred = pred_exp[i]

        for j in range(num_explanations):

            unpadded_traces = remove_padding_np(true_exp[j],unk_ent_id,unk_rel_id)

            unpadded_weights = true_weight[j][true_weight[j] != unk_weight_id]

            indices = (unpadded_traces == current_pred).all(axis=1)

            sum_weights = sum([float(num) for num in unpadded_weights[indices]])

            relevance_scores[j] += sum_weights

    precision_scores = relevance_scores / (n * .9)
    recall_scores = relevance_scores /  (num_true_triples * .9)

    nonzero_indices = (precision_scores + recall_scores) != 0

    if np.sum(nonzero_indices) == 0:
        f1_scores = [0.0]
    else:

        nonzero_precision_scores = precision_scores[nonzero_indices]
        nonzero_recall_scores = recall_scores[nonzero_indices]

        f1_scores = 2 * (nonzero_precision_scores * \
            nonzero_recall_scores) / (nonzero_precision_scores + nonzero_recall_scores)

    #f1_scores = 2 * (precision_scores * recall_scores) / (precision_scores + recall_scores + .000001)

    f1 = np.max(f1_scores)
    precision = np.max(precision_scores)
    recall = np.max(recall_scores)

    return precision, recall, f1

def pad_trace(trace,longest_trace,max_padding,unk):

    #unk = np.array([['UNK_ENT','UNK_REL','UNK_ENT']])
    
    unk = np.repeat(unk,[max_padding],axis=0)
    
    unk = np.expand_dims(unk,axis=0)
    
    while trace.shape[0] != longest_trace:
        trace = np.concatenate([trace,unk],axis=0)
        
    return trace

def pad_weight(trace,longest_trace,unk_weight):

    while trace.shape[0] != longest_trace:
        trace = np.concatenate([trace,unk_weight],axis=0)

    return trace

def f1(precision,recall):
    return 2 * (precision*recall) / (precision + recall)

def jaccard_score_np(true_exp,pred_exp):
        
    num_true_traces = true_exp.shape[0]
    num_pred_traces = pred_exp.shape[0]

    count = 0
    for pred_row in pred_exp:
        for true_row in true_exp:
            if (pred_row == true_row).all():
                count +=1

    score = count / (num_true_traces + num_pred_traces-count)
    
    return score

def jaccard_score_py(true_exp,pred_exp):

    num_true_traces = true_exp.shape[0]
    num_pred_traces = pred_exp.shape[0]

    count = 0
    for i in range(num_pred_traces):

        pred_row = pred_exp[i]

        for j in range(num_true_traces):

            true_row = true_exp[j]

            count += torch.where(torch.all(pred_row == true_row), 1,  0)

    score = count / (num_true_traces + num_pred_traces-count)
    
    return score

def remove_padding_np(exp,unk_ent_id, unk_rel_id,axis=1):

    #unk = np.array(['UNK_ENT', 'UNK_REL', 'UNK_ENT'])
    unk = np.array([unk_ent_id, unk_rel_id, unk_ent_id],dtype=object)

    exp_mask = (exp != unk).all(axis=axis)

    masked_exp = exp[exp_mask]

    return masked_exp

def remove_padding_py(exp,unk_ent_id, unk_rel_id,axis=-1):

    unk = torch.tensor([unk_ent_id, unk_rel_id, unk_ent_id],dtype=exp.dtype)

    exp_mask = torch.all(torch.ne(exp, unk),axis=axis)

    masked_exp = torch.masked_select(exp,exp_mask)

    return masked_exp

def max_jaccard_np(current_traces,pred_exp,true_weight,
    unk_ent_id,unk_rel_id,unk_weight_id,return_idx=False):

    ''''
    pred_exp must have shape[0] >= 1

    pred_exp: 2 dimensional (num_triples,3)

    '''
    
    jaccards = []
    sum_weights = []
    
    for i in range(len(current_traces)):
        
        true_exp = remove_padding_np(current_traces[i],unk_ent_id,unk_rel_id)

        weight = true_weight[i][true_weight[i] != unk_weight_id]

        sum_weight = sum([float(num) for num in weight])

        sum_weights.append(sum_weight)

        jaccard = jaccard_score_np(true_exp, pred_exp)

        jaccards.append(jaccard)

    max_indices = np.array(jaccards) == max(jaccards)

    if max_indices.sum() > 1:
        max_idx = np.argmax(max_indices * sum_weights)
        max_jaccard = jaccards[max_idx]
    else:
        max_jaccard = max(jaccards)
        max_idx = np.argmax(jaccards)
    
    if return_idx:
        return max_jaccard, max_idx
    return max_jaccard

def max_jaccard_py(current_traces,pred_exp,unk_ent_id,unk_rel_id):

    '''pred_exp: 2 dimensional (num_triples,3)'''
    
    jaccards = []
    
    for i in range(len(current_traces)):
        
        trace = remove_padding_py(current_traces[i],unk_ent_id,unk_rel_id)

        jaccard = jaccard_score_py(trace, pred_exp)

        jaccards.append(jaccard)

    return max(jaccards)

def parse_ttl(file_name, max_padding):
    
    lines = []

    with open(file_name, 'r') as f:
        for line in f:
            lines.append(line)

    ground_truth = []
    traces = []
    weights = []

    for idx in range(len(lines)):

        if "graph us:construct" in lines[idx]:

            split_source = lines[idx+1].split()

            source_rel = split_source[1].split(':')[1]

            source_tup = [split_source[0],source_rel,split_source[2]]

            weight = float(lines[idx+2].split()[2][1:5])

        exp_triples = []

        if 'graph us:where' in lines[idx]:

            while lines[idx+1] != '} \n':

                split_exp = lines[idx+1].split()

                exp_rel = split_exp[1].split(':')[1]

                exp_triple = [split_exp[0],exp_rel,split_exp[2]]

                exp_triples.append(exp_triple)

                idx+=1

        if len(source_tup) and len(exp_triples):

            if len(exp_triples) < max_padding:

                while len(exp_triples) != max_padding:

                    pad = np.array(['UNK_ENT', 'UNK_REL', 'UNK_ENT'])
                    exp_triples.append(pad)

            ground_truth.append(np.array(source_tup))
            traces.append(np.array(exp_triples))
            weights.append(weight)
            
    return np.array(ground_truth),np.array(traces),np.array(weights)

def get_data(data,rule):

    if rule == 'full_data':

        triples = data['all_triples']
        traces = data['all_traces'] 
        weights = data['all_weights']

        entities = data['all_entities'].tolist()
        relations = data['all_relations'].tolist()
    
    elif rule == 'test_full':

        triples,traces,weights,entities, relations = concat_triples_new(data, ['child', 'grandparent', 'parent', 'spouse'],4940)
        # triples,traces,weights,entities, relations = concat_triples_new(data, ['child', 'grandparent', 'parent', 'spouse'],2500)

    else:
        triples,traces,weights = concat_triples(data, [rule])
        entities = data[rule+'_entities'].tolist()
        relations = data[rule+'_relations'].tolist()

    return triples,traces,weights,entities,relations

def concat_triples_new(data, rules, size=None):

    triples = []
    traces = []
    weights = []
    np.random.seed(123)
    for rule in rules:

        # triple_name = rule + '_triples'
        # traces_name = rule + '_traces'
        # weights_name = rule + '_weights'
        idx = (data['all_triples'][:,1]==rule)
        if size!=None:  #改成随机的
            iidx = np.random.choice(np.arange(idx.sum()),size=size, replace=False)
            triples.append(data['all_triples'][idx][iidx])
            traces.append(data['all_traces'][idx][iidx])
            weights.append(data['all_weights'][idx][iidx])
            # triples.append(data['all_triples'][idx][:size])
            # traces.append(data['all_traces'][idx][:size])
            # weights.append(data['all_weights'][idx][:size])
        else:
            triples.append(data['all_triples'][idx])
            traces.append(data['all_traces'][idx])
            weights.append(data['all_weights'][idx])

    triples = np.concatenate(triples, axis=0)
    traces = np.concatenate(traces, axis=0)
    weights = np.concatenate(weights,axis=0)

    entities = np.unique(np.concatenate([triples[:,0],triples[:,2],traces.reshape(-1,3)[:,0],traces.reshape(-1,3)[:,2]]))
    relations = ['UNK_REL','child', 'grandparent', 'parent', 'spouse','brother','sister']
    
    return triples, traces, weights,entities, relations

def concat_triples(data, rules):

    triples = []
    traces = []
    weights = []

    for rule in rules:

        triple_name = rule + '_triples'
        traces_name = rule + '_traces'
        weights_name = rule + '_weights'

        triples.append(data[triple_name])
        traces.append(data[traces_name])
        weights.append(data[weights_name])

    triples = np.concatenate(triples, axis=0)
    traces = np.concatenate(traces, axis=0)
    weights = np.concatenate(weights,axis=0)
    
    return triples, traces, weights

def array2idx(dataset,ent2idx,rel2idx):
    
    if dataset.ndim == 2:
        
        data = []
        
        for head, rel, tail in dataset:
            
            head_idx = ent2idx[head]
            tail_idx = ent2idx[tail]
            rel_idx = rel2idx[rel]
            
            data.append((head_idx, rel_idx, tail_idx))

        data = np.array(data)

    elif dataset.ndim == 3:
        
        data = []

        for i in range(len(dataset)):
            
            temp_array = []
        
            for head,rel,tail in dataset[i,:,:]:

                head_idx = ent2idx[head]
                tail_idx = ent2idx[tail]
                rel_idx = rel2idx[rel]

                temp_array.append((head_idx,rel_idx,tail_idx))

            data.append(temp_array)
            
        data = np.array(data).reshape(-1,dataset.shape[1],3)

    elif dataset.ndim == 4:

        data = []

        for i in range(len(dataset)):

            temp_array = []

            for j in range(len(dataset[i])):

                temp_array_1 = []

                for head,rel,tail in dataset[i,j]:

                    head_idx = ent2idx[head]
                    tail_idx = ent2idx[tail]
                    rel_idx = rel2idx[rel]

                    temp_array_1.append((head_idx,rel_idx,tail_idx))

                temp_array.append(temp_array_1)

            data.append(temp_array)

        data = np.array(data)

    return data

def idx2array(dataset,idx2ent,idx2rel):
    
    if dataset.ndim == 2:
        
        data = []
        
        for head_idx, rel_idx, tail_idx in dataset:
            
            head = idx2ent[head_idx]
            tail = idx2ent[tail_idx]
            rel = idx2rel[rel_idx]
            
            data.append((head, rel, tail))

        data = np.array(data)

    elif dataset.ndim == 3:
        
        data = []

        for i in range(len(dataset)):
            
            temp_array = []
        
            for head_idx, rel_idx, tail_idx in dataset[i,:,:]:

                head = idx2ent[head_idx]
                tail = idx2ent[tail_idx]
                rel = idx2rel[rel_idx]

                temp_array.append((head,rel,tail))

            data.append(temp_array)
            
        data = np.array(data).reshape(-1,dataset.shape[1],3)

    elif dataset.ndim == 4:

        data = []

        for i in range(len(dataset)):

            temp_array = []

            for j in range(len(dataset[i])):

                temp_array_1 = []

                for head_idx, rel_idx, tail_idx in dataset[i,j]:

                    head = idx2ent[head_idx]
                    tail = idx2ent[tail_idx]
                    rel = idx2rel[rel_idx]

                    temp_array_1.append((head,rel,tail))

                temp_array.append(temp_array_1)

            data.append(temp_array)

        data = np.array(data)

    return data

def distinct(a):
    _a = torch.unique(a,dim=0)
    return _a

def get_adj_mats(data,num_entities,num_relations):

    adj_mats = []

    for i in range(num_relations):

        data_i = torch.tensor(data[data[:,1] == i])

        if not data_i.shape[0]:
            indices = torch.zeros((1,2),dtype=torch.int64)
            values = torch.zeros((indices.shape[0]))

        else:
            # indices = torch.gather(data_i,dim=1,index=torch.tensor([0,2]))
            indices = data_i[:,[0,2]]

            # indices = tf.py_function(distinct,[indices],indices.dtype)
            indices = distinct(indices)
            values = torch.ones((indices.shape[0]))

        sparse_mat = torch.sparse.FloatTensor(
            indices=indices.permute(1,0),
            values=values,
            size=(num_entities,num_entities)
            )
        sparse_mat.unsqueeze(dim=0)
        # sparse_mat = tf.sparse.reorder(sparse_mat)
        # sparse_mat = tf.sparse.reshape(sparse_mat, shape=(1,num_entities,num_entities))

        adj_mats.append(sparse_mat)

    return adj_mats

def get_negative_triples(head, rel, tail, num_entities, random_state=123):
    torch.manual_seed(random_state)
    cond = torch.empty(head.shape, dtype=torch.int64).random_(0, 2)
    rnd = torch.empty(head.shape, dtype=torch.int64).random_(0, num_entities-1)
    # cond = tf.random.uniform(tf.shape(head), 0, 2, dtype=torch.int64, seed=random_state)
    # rnd = tf.random.uniform(tf.shape(head), 0, num_entities-1, dtype=torch.int64, seed=random_state)
    
    neg_head = torch.where(cond == 1, head, rnd)
    neg_tail = torch.where(cond == 1, rnd, tail)

    return neg_head, neg_tail
    
def train_test_split_no_unseen(
    X,
    E,
    weights=None,
    longest_trace=None,
    max_padding=None,
    unk_ent_id='UNK_ENT',
    unk_rel_id='UNK_REL',
    test_size=.25,
    seed=123,
    allow_duplication=False):

    test_size = int(len(X) * test_size)

    np.random.seed(seed)

    X_train = None
    X_train_exp = None
    X_test_candidates = X
    X_test_exp_candidates = E

    if E.ndim == 4:

        exp_entities = np.array([
            [E[:,i,j,0],E[:,i,j,2]] for i in range(longest_trace) for j in range(max_padding)]).flatten()

        exp_relations = np.array([
            [E[:,i,j,1]] for i in range(longest_trace) for j in range(max_padding)]).flatten()
        
    elif E.ndim == 3:
        exp_entities = np.array([[E[:,i,:][:,0],E[:,i,:][:,2]] for i in range(max_padding)]).flatten()

        exp_relations = np.array([E[:,i,:][:,1] for i in range(max_padding)]).flatten()
        
    entities, entity_cnt = np.unique(np.concatenate([
                                X[:,0], X[:,2], exp_entities],axis=0),return_counts=True)
    rels, rels_cnt = np.unique(np.concatenate([
                                X[:,1], exp_relations],axis=0),return_counts=True)
    
    dict_entities = dict(zip(entities, entity_cnt))
    dict_rels = dict(zip(rels, rels_cnt))
    idx_test = []
    idx_train = []
    
    all_indices_shuffled = np.random.permutation(np.arange(X_test_candidates.shape[0]))

    for i, idx in enumerate(all_indices_shuffled):
        test_triple = X_test_candidates[idx]
        test_exp = remove_padding_np(X_test_exp_candidates[idx],unk_ent_id, unk_rel_id,axis=-1)
                
        # reduce the entity and rel count of triple
        dict_entities[test_triple[0]] = dict_entities[test_triple[0]] - 1
        dict_rels[test_triple[1]] = dict_rels[test_triple[1]] - 1
        dict_entities[test_triple[2]] = dict_entities[test_triple[2]] - 1
        
        exp_entities = np.concatenate([test_exp[:,0].flatten(),
                                       test_exp[:,2].flatten()])
        
        exp_rels = test_exp[:,1]
        
        # reduce the entity and rel count of explanation
        for exp_ent in exp_entities:
            dict_entities[exp_ent] -= 1
            
        for exp_rel in exp_rels:
            dict_rels[exp_rel] -= 1
            
        ent_counts = []
        for exp_ent in exp_entities:
            count_i = dict_entities[exp_ent]
            
            if count_i > 0:
                ent_counts.append(1)
            else:
                ent_counts.append(0)
                
        rel_counts = []
        for exp_rel in exp_rels:
            count_i = dict_rels[exp_rel]
            
            if count_i > 0:
                rel_counts.append(1)
            else:
                rel_counts.append(0)
        
        #compute sums and determine if counts > 0

        # test if the counts are > 0
        if dict_entities[test_triple[0]] > 0 and \
                dict_rels[test_triple[1]] > 0 and \
                dict_entities[test_triple[2]] > 0 and \
                sum(ent_counts) == len(ent_counts) and \
                sum(rel_counts) == len(rel_counts):
            
            # Can safetly add the triple to test set
            idx_test.append(idx)
            if len(idx_test) == test_size:
                # Since we found the requested test set of given size
                # add all the remaining indices of candidates to training set
                idx_train.extend(list(all_indices_shuffled[i + 1:]))
                
                # break out of the loop
                break
            
        else:
            # since removing this triple results in unseen entities, add it to training
            dict_entities[test_triple[0]] = dict_entities[test_triple[0]] + 1
            dict_rels[test_triple[1]] = dict_rels[test_triple[1]] + 1
            dict_entities[test_triple[2]] = dict_entities[test_triple[2]] + 1
            
            for exp_ent in exp_entities:
                dict_entities[exp_ent] += 1
            
            for exp_rel in exp_rels:
                dict_rels[exp_rel] += 1
            
            idx_train.append(idx)
            
    if len(idx_test) != test_size:
        # if we cannot get the test set of required size that means we cannot get unique triples
        # in the test set without creating unseen entities
        if allow_duplication:
            # if duplication is allowed, randomly choose from the existing test set and create duplicates
            duplicate_idx = np.random.choice(idx_test, size=(test_size - len(idx_test))).tolist()
            idx_test.extend(list(duplicate_idx))
        else:
            # throw an exception since we cannot get unique triples in the test set without creating 
            # unseen entities
            raise Exception("Cannot create a test split of the desired size. "
                            "Some entities will not occur in both training and test set. "
                            "Set allow_duplication=True," 
                            "or set test_size to a smaller value.")

    X_train = X_test_candidates[idx_train]
    X_train_exp = X_test_exp_candidates[idx_train]
    
    X_test = X_test_candidates[idx_test]
    X_test_exp = X_test_exp_candidates[idx_test]
    
    #shuffle data
    
    idx_train_shuffle = np.random.permutation(np.arange(len(idx_train)))
    idx_test_shuffle = np.random.permutation(np.arange(len(idx_test)))
    
    X_train = X_train[idx_train_shuffle]
    X_train_exp = X_train_exp[idx_train_shuffle]
    
    X_test = X_test[idx_test_shuffle]
    X_test_exp = X_test_exp[idx_test_shuffle]
    
    if weights is not None:
        
        X_train_weights = weights[idx_train]
        X_test_weights = weights[idx_test]
        
        X_train_weights = X_train_weights[idx_train_shuffle]
        X_test_weights = X_test_weights[idx_test_shuffle]
                
        return X_train, X_train_exp,X_train_weights,\
               X_test, X_test_exp, X_test_weights
    
    return X_train, X_train_exp, X_test, X_test_exp
