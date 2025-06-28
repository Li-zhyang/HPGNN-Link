import numpy as np
import os.path as osp
import utils
import torch
from torch_geometric.utils import to_dense_adj, remove_self_loops
np.random.seed(123)

def load_data_fr(data_dir, dataset_name, rule, seed, splits):
    data = np.load(osp.join(data_dir, dataset_name + '.npz'))
    triples,traces,weights,entities,relations = utils.get_data(data, rule)

    MAX_PADDING = 2
    LONGEST_TRACE = utils.get_longest_trace(data, rule)

    X_train_triples, X_train_traces, X_train_weights, X_test_triples, X_test_traces, X_test_weights = utils.train_test_split_no_unseen(
        X=triples,E=traces,weights=weights,
        longest_trace=LONGEST_TRACE,max_padding=MAX_PADDING,
        test_size=splits['test'],seed=seed)

    NUM_ENTITIES = len(entities)
    NUM_RELATIONS = len(relations)

    ent2idx = dict(zip(entities, range(NUM_ENTITIES)))
    rel2idx = dict(zip(relations, range(NUM_RELATIONS)))

    idx2ent = dict(zip(range(NUM_ENTITIES),entities))

    triples2idx = utils.array2idx(triples, ent2idx, rel2idx)
    traces2idx = utils.array2idx(traces, ent2idx, rel2idx)

    X_train = np.concatenate([X_train_triples, X_train_traces.reshape(-1,3)],axis=0)
    idx = np.sort(np.unique(X_train,axis=0,return_index=True)[1])
    X_train = X_train[idx]
    X_train = utils.array2idx(X_train, ent2idx, rel2idx)

    X_test = np.concatenate([X_test_triples, X_test_traces.reshape(-1,3)],axis=0)
    idx = np.sort(np.unique(X_test,axis=0,return_index=True)[1])
    X_test = X_test[idx]
    X_test = utils.array2idx(X_test, ent2idx, rel2idx)

    X_all = np.concatenate([X_train, X_test],axis=0)
    
    X_train_triples = utils.array2idx(X_train_triples, ent2idx, rel2idx)
    X_train_traces = utils.array2idx(X_train_traces, ent2idx, rel2idx)
    X_test_triples = utils.array2idx(X_test_triples, ent2idx, rel2idx)
    X_test_traces = utils.array2idx(X_test_traces, ent2idx, rel2idx)

    X_test_graph = np.concatenate([X_train, X_test_traces.reshape(-1,3)],axis=0)
    X_test_graph = np.unique(X_test_graph,axis=0)

    dataset = {'X_train':X_train, 'X_train_triples':X_train_triples, 'X_train_traces':X_train_traces, 'X_train_weights':X_train_weights, 'X_test_triples':X_test_triples,
                'X_test':X_test, 'X_test_traces':X_test_traces, 'X_test_weights':X_test_weights, 'ent2idx':ent2idx, 'rel2idx':rel2idx, 
                'triples2idx':triples2idx, 'traces2idx':traces2idx, 'X_all':X_all, 'X_test_graph':X_test_graph}

    return dataset

def neg_sampling(pos_samples, negative_rate, num_entity, num):    
    neg_num = int(num * negative_rate)
    values = torch.randint(0,num_entity,(neg_num,))
    choices = torch.rand(neg_num)
    subj = choices > 0.5
    obj = choices <= 0.5
    neg_samples = pos_samples[:neg_num,:]
    neg_samples[subj, 0] = values[subj]
    neg_samples[obj, 2] = values[obj]
    neg_samples[:,1] = torch.zeros(neg_samples.shape[0])
    return neg_samples


def remove_loop_and_same(triples, unk_ent=None):
    new_triples=triples[0].unsqueeze(0)
    tmp = set()
    tmp.add((triples[0][0].item(),triples[0][2].item()))
    idx = [0]
    i=0
    for line in triples:
        if unk_ent==None:
            if (line[0].item(),line[2].item()) not in tmp and line[0].item()!=line[2].item():
                new_triples = torch.cat([new_triples,line.unsqueeze(0)],dim=0)
                tmp.add((line[0].item(),line[2].item()))
                idx.append(i)
        else:
            if (line[0].item(),line[2].item()) not in tmp and line[0].item()!=line[2].item() and line[0]!=unk_ent and line[2]!=unk_ent:
                new_triples = torch.cat([new_triples,line.unsqueeze(0)],dim=0)
                tmp.add((line[0].item(),line[2].item()))
                idx.append(i)
        i+=1

    return new_triples,idx

def build_graph_fr(nodes, rels, triples, links, negative_rate, traces, weights):
    num_nodes = len(nodes)
    num_rels = len(rels)
    triples = torch.from_numpy(triples)
    neg_samples = neg_sampling(triples.clone(), negative_rate, num_nodes, links.shape[0])
    triples = torch.cat([triples,neg_samples], dim=0)
    triples,_ = remove_loop_and_same(triples, nodes['UNK_ENT'])

    src, rel, dst = triples.T
    rel = rel

    edge_index = torch.stack((src, dst))
    edge_type = rel
    edge_index, edge_type = remove_self_loops(edge_index, edge_type)
    edge_type_tmp = edge_type

    adj = to_dense_adj(edge_index, edge_attr=edge_type_tmp, max_num_nodes = len(nodes))


    triples_link = torch.cat([torch.from_numpy(links),neg_samples], dim=0)
    triples_tmp,idx = remove_loop_and_same(triples_link, nodes['UNK_ENT'])
    src, rel, dst = triples_tmp.T
    rel = rel
    edge_index = torch.stack((src, dst))
    edge_type = rel

    weights = np.where(weights == 'UNK_WEIGHT', 0, weights).astype(np.float32)
    weights = torch.from_numpy(weights)

    traces = torch.from_numpy(traces)
    neg_traces = torch.tensor([nodes['UNK_ENT'],rels['UNK_REL'],nodes['UNK_ENT']]).repeat(triples_link.shape[0]-links.shape[0],traces.shape[1],traces.shape[2],1)
    neg_weights = torch.tensor([0,0]).repeat(triples_link.shape[0]-links.shape[0],weights.shape[1],1) 
    traces = traces.reshape(traces.shape[0],-1,3).transpose(1,2)
    neg_traces = neg_traces.reshape(neg_traces.shape[0],-1,3).transpose(1,2)
    weights = weights.reshape(weights.shape[0], -1, 1).transpose(1,2)
    neg_weights = neg_weights.reshape(neg_weights.shape[0], -1, 1).transpose(1,2)
    traces = torch.cat([traces,neg_traces],dim=0)[idx]
    weights = torch.cat([weights,neg_weights],dim=0)[idx]

    return adj, edge_index, edge_type, traces, weights

def load_txt(triples_path):
    textual_triples = []

    with open(triples_path, "r") as triples_file:
        lines = triples_file.readlines()
        for line in lines:
            head_name, relation_name, tail_name = line.strip().split('\t')

            head_name = head_name.replace(",", "").replace(":", "").replace(";", "")
            relation_name = relation_name.replace(",", "").replace(":", "").replace(";", "")
            tail_name = tail_name.replace(",", "").replace(":", "").replace(";", "")

            textual_triples.append([head_name, relation_name, tail_name])

    return np.array(textual_triples)

def load_data_w(data_dir):
    train_data = load_txt(osp.join(data_dir, 'train.txt'))#'./WN18RR/train.txt')#np.load(osp.join(data_dir, dataset_name + '.npz'))
    test_data = load_txt(osp.join(data_dir, 'test.txt'))#'./WN18RR/test.txt')
    data = np.concatenate([train_data,test_data],axis=0)
    entities = np.unique(np.concatenate([data[:,0],data[:,2]]))
    relations = np.unique(data[:,1])

    NUM_ENTITIES = len(entities)
    NUM_RELATIONS = len(relations)

    ent2idx = dict(zip(entities, range(NUM_ENTITIES)))
    rel2idx = dict(zip(relations, range(1,NUM_RELATIONS+1)))

    X_train = utils.array2idx(train_data, ent2idx, rel2idx)
    X_test = utils.array2idx(test_data, ent2idx, rel2idx)

    X_train_triples = []
    X_test_triples = []
    for i in range(1,NUM_RELATIONS+1):
        ridx = (X_train[:,1]==i)
        NUM_SAMPLES = (X_test[:,1]==i).sum()*4
        idx = np.random.choice(range(ridx.sum()), NUM_SAMPLES if NUM_SAMPLES<=ridx.sum() else ridx.sum(), replace=False)
        X_train_triples.extend(X_train[ridx][idx])
    X_train_triples = np.array(X_train_triples)
    X_test_triples = X_test

    dataset = {'X_train':X_train, 'X_train_triples':X_train_triples, 'X_test_triples':X_test_triples, 'ent2idx':ent2idx, 'rel2idx':rel2idx}

    return dataset

def build_graph_w(nodes, rels, triples, links, negative_rate):
    num_nodes = len(nodes)
    num_rels = len(rels)
    triples = torch.from_numpy(triples)
    neg_samples = neg_sampling(triples.clone(), negative_rate, num_nodes, links.shape[0])
    triples = torch.cat([triples,neg_samples], dim=0)
    triples,_ = remove_loop_and_same(triples)

    src, rel, dst = triples.T
    rel = rel

    edge_index = torch.stack((src, dst))
    edge_type = rel
    edge_index, edge_type = remove_self_loops(edge_index, edge_type)
    edge_type_tmp = edge_type

    adj = to_dense_adj(edge_index, edge_attr=edge_type_tmp, max_num_nodes = len(nodes))


    triples_link = torch.cat([torch.from_numpy(links),neg_samples], dim=0)
    triples_tmp,idx = remove_loop_and_same(triples_link)
    src, rel, dst = triples_tmp.T
    rel = rel
    edge_index = torch.stack((src, dst))
    edge_type = rel

    return adj, edge_index, edge_type

def load_txt1(triples_path):
    textual_triples = []

    with open(triples_path, "r") as triples_file:
        lines = triples_file.readlines()
        for line in lines:
            if len(line.strip().split(' '))==4:
                if line.strip().split(' ')[-1]=='0':
                    continue
            relation_name, head_name, tail_name = line.strip().split(' ')[:3]

            head_name = head_name.replace(",", "").replace(":", "").replace(";", "")
            relation_name = relation_name.replace(",", "").replace(":", "").replace(";", "")
            tail_name = tail_name.replace(",", "").replace(":", "").replace(";", "")

            textual_triples.append([head_name, relation_name, tail_name])

    return np.array(textual_triples)

def load_data_a(data_dir):
    train_data = load_txt1(osp.join(data_dir, 'train.txt'))#'./WN18RR/train.txt')#np.load(osp.join(data_dir, dataset_name + '.npz'))
    test_data = load_txt1(osp.join(data_dir, 'test.txt'))#'./WN18RR/test.txt')
    data = np.concatenate([train_data,test_data],axis=0)
    entities = np.unique(np.concatenate([data[:,0],data[:,2]]))
    relations = np.unique(data[:,1])

    NUM_ENTITIES = len(entities)
    NUM_RELATIONS = len(relations)

    ent2idx = dict(zip(entities, range(NUM_ENTITIES)))
    rel2idx = dict(zip(relations, range(1,NUM_RELATIONS+1)))

    X_train = utils.array2idx(train_data, ent2idx, rel2idx)
    X_test = utils.array2idx(test_data, ent2idx, rel2idx)

    X_train_triples = X_train
    X_test_triples = X_test

    dataset = {'X_train':X_train, 'X_train_triples':X_train_triples, 'X_test_triples':X_test_triples, 'ent2idx':ent2idx, 'rel2idx':rel2idx}

    return dataset
    
