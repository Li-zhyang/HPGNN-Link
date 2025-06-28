import os,sys
import shutil
os.chdir(sys.path[0])
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from dataset_pg import MyDataset_pg
from dataset_fr import MyDataset_fr
from dataset_w import MyDataset_w
from tqdm import tqdm
from model.RGCN import RGCN
from model.generator_path import Generator
from model.hpgnn import HPGNN
import torch
from torch_geometric.loader import DataLoader
from load_data import load_data_fr, build_graph_fr, load_data_w, build_graph_w, load_data_a
from torch_geometric.utils import to_dense_adj
from graph_utils import unbatch, unbatch_edge_index
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import networkx as nx
from data_processing import load_dataset_pg, load_dataset_pg1
import torch.nn.functional as F
device = torch.device("cuda:0")

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="HPGNN Training Script")
    parser.add_argument('--dataset', type=str, default='french_royalty', choices=['french_royalty', 'european_royalty', 'synthetic', 'aug_citation', 'WN18RR', 'Amazon'],
                        help="Dataset name (default: french_royalty)")
    parser.add_argument('--batch_size', type=int, default=64,
                        help="Batch size for training (default: 64)")
    parser.add_argument('--epochs', type=int, default=2000,
                        help="Number of epochs for training (default: 2000)")
    parser.add_argument('--lr', type=float, default=0.0001,
                        help="Initial learning rate (default: 0.0001)")
    parser.add_argument('--dropout', type=float, default=0.05,
                        help="Dropout rate (default: 0.05)")
    parser.add_argument('--seed', type=int, default=123,
                        help="Random seed (default: 123)")
    parser.add_argument('--max_length', type=int, default=2,
                        help="Maximum length (default: 2)")
    parser.add_argument('--negative_ratio', type=float, default=0.13,
                        help="Negative ratio (default: 0.13)")
    parser.add_argument('--k_core', type=int, default=3,
                        help="K-core parameter (default: 3)")
    parser.add_argument('--num_bases', type=int, default=4,
                        help="Number of bases for RGCN (default: 4)")
    parser.add_argument('--sample_ratio', type=float, default=1.0,
                        help="Sample ratio (default: 1.0)")
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help="Weight decay (default: 5e-4)")
    parser.add_argument('--max_num_nodes', type=int, default=200,
                        help="Maximum number of nodes (default: 200)")
    parser.add_argument('--K', type=int, default=5,
                        help="Number of prototypes each class (default: 5)")
    parser.add_argument('--nhid', type=int, default=128,
                        help="Hidden size (default: 128)")
    parser.add_argument('--alpha', type=float, default=1.0,
                    help="Weight coefficient for the information loss (default: 1.0)")
    parser.add_argument('--beta', type=float, default=1.0,
                    help="Weight coefficient for the prototype loss (default: 1.0)")
    
    args = parser.parse_args()
    return args

args = parse_args()

torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
random_state = args.seed

batch_size = args.batch_size
# batch_size = 8 #FR=64, ER=32, aug=16, syn=128, W=64

k_core = args.k_core    #aug=3,syn=2
dataset_name = args.dataset #'synthetic'  'aug_citation'  'french_royalty' 'european_royalty' 'WN18RR' 'Amazon'
max_length = args.max_length # FR=ER=2,other=3

negative_ratio = args.negative_ratio     
# negative_ratio = 0.16     #FR
# negative_ratio = 0.25     #ER
# negative_ratio = 0.091    #W
# negative_ratio = 0.5      #Amazon

epochs = args.epochs
lr=args.lr
nhid=args.nhid
dropout=args.dropout
alpha = args.alpha
beta = args.beta
K = args.K
num_bases = args.num_bases
sample_ratio=args.sample_ratio
weight_decay=args.weight_decay
max_num_nodes=args.max_num_nodes

splits={'test': 0.2}
k=2
lr1=0.0005
lr2=0.0005
split_ratio = 1
grad_norm = 1


def train_one_epoch(hpgnn, dataloader, optimizer, device, split_rate, alpha, H_init, epoch, class_weight):
    hpgnn.train()
    # pbar = tqdm(dataloader)
    pbar = dataloader
    train_loss = []
    accuracy = []
    accuracy1 = []
    accuracy2 = []
    yloss = []
    rloss = []
    infoloss = []
    pathloss = []
    l2loss = []
    for idx, data in enumerate(pbar):
        data = data.to(device)
        if epoch>=1000:
            hpgnn.activate_path(True)
        
        predict_y, _, _, dist, L_info, L_path, L_l2 = hpgnn.forward(data.x, data.edge_index, data.edge_type, data.batch, data.links, data.s, data.t, data.hyper_edge, data.edge_gt_att)
        
        L_y = hpgnn.c_loss(predict_y, data.y, class_weight)
        L_c = hpgnn.cls_loss(dist, data.y)

        loss = L_y + alpha*L_info + beta*L_c

        if epoch>=1000:
            loss += 0.001*L_path + 0.001*L_l2
            hpgnn.activate_path(False)

        yloss.append(L_y.detach())
        rloss.append(L_c.detach())
        infoloss.append(L_info.detach())
        train_loss.append(loss.detach())
        l2loss.append(L_l2.detach())
        pathloss.append(L_path.detach())
        acc,acc1,acc2 = hpgnn.acc(predict_y, data.y)
        accuracy.append(acc)
        accuracy1.append(acc1)
        accuracy2.append(acc2)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(hpgnn.parameters(), max_norm=1.0)
        optimizer.step()
        data = data.to('cpu')
    return sum(train_loss)/len(train_loss), sum(accuracy)/len(accuracy), sum(yloss)/len(yloss), sum(rloss)/len(rloss), sum(accuracy1)/len(accuracy1), sum(accuracy2)/len(accuracy2), sum(infoloss)/len(infoloss), sum(pathloss)/len(pathloss), sum(l2loss)/len(l2loss)

def project_one_epoch(hpgnn, dataset, device):
    hpgnn.eval()
    for i in range(hpgnn.num_class):
        pbar = DataLoader(dataset.get_by_label(i), batch_size, shuffle=True, drop_last=False)
        for j in range(hpgnn.K):
            max_sim = 0.0
            max_hg_pro = torch.empty(0)
            for idx, data in enumerate(pbar):
                data = data.to(device)
                hg_pro, sim = hpgnn.project_forward(data.x, data.edge_index, data.edge_type, data.batch, data.links, hpgnn.H_init[i][j], data.hyper_edge)
                if sim>max_sim:
                    max_sim = sim
                    max_hg_pro = hg_pro
                data = data.to('cpu')
            hpgnn.H_init[i][j]=max_hg_pro

def visualize_a_graph(idx, edge_index, edge_att, edge_type, link, select=True, val = 0.5):
    plt.clf()
    plt.figure(dpi=300)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    G = nx.DiGraph()
    edge_index = edge_index.transpose(0,1)
    G_node = nx.Graph()
    G_node.add_node(link[0])
    G_node.add_node(link[1])
    G_node.add_edge(link[0],link[1])

    for i in range(len(edge_index)):
        source, target = edge_index[i]
        etype = edge_type[i].item()
        weight = edge_att[i].item()
        if select:
            if weight>val:
                G.add_edge(source.item(), target.item(), edge_weight=weight, edge_type=etype)
        else:
            G.add_edge(source.item(), target.item(), edge_weight=weight, edge_type=etype)


    pos = nx.kamada_kawai_layout(G)
    edge_colors_tmp = [data['edge_weight'] for src, dst, data in G.edges(data=True)]
    if select:
        edge_colors = [0.1+0.9*(value-min(edge_colors_tmp))/(max(edge_colors_tmp)-min(edge_colors_tmp)+1e-6) for value in edge_colors_tmp]
    else:
        edge_colors = [0.1+0.9*(value-0)/(1-0+1e-6) for value in edge_colors_tmp]
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=300)
    nx.draw_networkx_nodes(G_node, pos, node_color='red', node_size=300)
    nx.draw_networkx_edges(G, pos, edge_color='black', alpha=edge_colors, 
                        width=2, arrows=True, connectionstyle="arc3,rad=0.1", arrowsize=10)
    nx.draw_networkx_edges(G_node, pos, edge_color='red', 
                        width=2, arrows=True, connectionstyle="arc3,rad=0.1", style='dashed', arrowsize=10)
    plt.show()
    plt.savefig('./save_fig/'+str(idx)+'.png')

def evaluate_pg(edge_index, sij, trace_edge_index, num_nodes, trace_edge_type, comp_num, edge_type):
    if not (sij==-1).all():
        Idj = to_dense_adj(edge_index, edge_attr=edge_type, max_num_nodes=num_nodes)[0]
        Adj = to_dense_adj(edge_index, edge_attr=sij.squeeze(1), max_num_nodes=num_nodes)[0]
        Edj = to_dense_adj(trace_edge_index, edge_attr = torch.ones(trace_edge_index.shape[1],dtype=torch.bool).to(device), max_num_nodes=num_nodes)[0]
        indices = torch.where((Idj.unsqueeze(2) == trace_edge_type).any(dim=2))

        y_score = Adj[indices].reshape(-1).cpu().detach()
        y_true = Edj[indices].reshape(-1).cpu().detach()
        roc_auc = roc_auc_score(y_true, y_score)
        y_score = torch.concat([y_score,torch.zeros(comp_num)])
        y_true = torch.concat([y_true,torch.zeros(comp_num)])
        return roc_auc, roc_auc_score(y_true, y_score)
    else:
        return 0.0

def explain_pg(predict_y, z, data, hpgnn, hg):
    claidx = predict_y.argmax(dim=1)
    x_unbatch = unbatch(z, data.batch)
    edge_index_unbatch, edge_type_unbatch = unbatch_edge_index(data.edge_index, data.batch, data.edge_type)
    trace_edge_index_unbatch, trace_edge_type_unbatch = unbatch_edge_index(data.trace_edge_index, data.batch, data.trace_edge_type)
    unbatched_data = data.to_data_list()
    rocauc = []
    jaccard = []
    for idx,i in zip(range(len(x_unbatch)),claidx):
        if idx<len(trace_edge_index_unbatch) and trace_edge_index_unbatch[idx].shape[1]!=0:
            H_emb = hpgnn.H_init[i]
            scores = torch.matmul(hg[idx], H_emb.T)
            att = F.softmax(scores, dim=-1)
            H_emb = torch.matmul(att, H_emb)

            edge_index_hat, sij = hpgnn.fg_model.get_explaination(x_unbatch[idx], H_emb, edge_index_unbatch[idx], edge_type_unbatch[idx], data.s[idx], data.t[idx], unbatched_data[idx].hyper_edge)
            roc_auc, ja = evaluate_pg(edge_index_unbatch[idx], sij, trace_edge_index_unbatch[idx], x_unbatch[idx].shape[0],trace_edge_type_unbatch[idx], data.comp_num[idx], edge_type_unbatch[idx])
            jaccard.append(ja)
            rocauc.append(roc_auc)
    return sum(rocauc)/len(rocauc) if len(rocauc)!=0 else None, sum(jaccard)/len(jaccard) if len(jaccard)!=0 else None

def evaluate_fr(edge_index_hat, trace_edge_index, weight, trace_edge_type):
    trace_edge_index = trace_edge_index.transpose(0,1).reshape(-1,2,2)
    weight = weight.reshape(-1,2,1)
    trace_edge_type = trace_edge_type.reshape(-1,2,1)
    edge_index_hat = edge_index_hat.transpose(0,1)
    n = len(edge_index_hat)
    max_ja = 0.0
    max_pre = 0.0
    max_re = 0.0
    max_f1 = 0.0
    for i in range(len(trace_edge_index)):
        same = 0
        true = 0
        count = 0
        for j in range(len(trace_edge_index[i])):
            if trace_edge_type[i][j]!=0:
                true+=1
                for k in range(n):
                    if (edge_index_hat[k]==trace_edge_index[i][j]).all():
                        same+=1*weight[i][j][0]
                        count+=1
        ja = count/(n+true-count)
        pre = same/(n*0.9) if n!=0 else 0.0
        re = same/(true*0.9) if true!=0 else 0.0
        f1 = 2*pre*re/(pre+re) if pre!=0 or re!=0 else 0.0
        max_ja = max_ja if max_ja>ja else ja
        max_pre = max_pre if max_pre>pre else pre
        max_re = max_re if max_re>re else re
        max_f1 = max_f1 if max_f1>f1 else f1
    return max_ja, max_pre, max_re, max_f1

def explain_fr(predict_y, z, data, hpgnn, hg):
    claidx = predict_y.argmax(dim=1)

    x_unbatch = unbatch(z, data.batch)
    edge_index_unbatch, edge_type_unbatch = unbatch_edge_index(data.edge_index, data.batch, data.edge_type)
    trace_edge_index_unbatch, trace_edge_type_unbatch = unbatch_edge_index(data.trace_edge_index, data.batch, data.trace_edge_type)
    _, trace_weight_unbatch = unbatch_edge_index(data.trace_edge_index, data.batch, data.trace_weight)
    unbatched_data = data.to_data_list()
    jaccard = []
    precision = []
    recall = []
    f1_score = []
    for idx,i in zip(range(len(x_unbatch)),claidx):
        if not (trace_edge_type_unbatch[idx]==0).all():
            H_emb = hpgnn.H_init[i]
            scores = torch.matmul(hg[idx], H_emb.T)
            att = F.softmax(scores, dim=-1)
            H_emb = torch.matmul(att, H_emb)
            edge_index_hat, sij = hpgnn.fg_model.get_explaination(x_unbatch[idx], H_emb, edge_index_unbatch[idx], edge_type_unbatch[idx], data.s[idx], data.t[idx], unbatched_data[idx].hyper_edge, True)
            if edge_index_hat.shape[1]==0:
                ja, p, r, f1 = torch.tensor(0.0,device=device),torch.tensor(0.0,device=device),torch.tensor(0.0,device=device),torch.tensor(0.0,device=device)
            else:
                ja, p, r, f1 = evaluate_fr(edge_index_hat, trace_edge_index_unbatch[idx], trace_weight_unbatch[idx], trace_edge_type_unbatch[idx])
            jaccard.append(ja)
            precision.append(p)
            recall.append(r)
            f1_score.append(f1)

    return sum(jaccard)/len(jaccard),sum(precision)/len(precision),sum(recall)/len(recall),sum(f1_score)/len(f1_score)

def explain_w(predict_y, z, data, hpgnn, hg):
    claidx = predict_y.argmax(dim=1)

    x_unbatch = unbatch(z, data.batch)

    edge_index_unbatch, edge_type_unbatch = unbatch_edge_index(data.edge_index, data.batch, data.edge_type)
    unbatched_data = data.to_data_list()
    sij_all = torch.empty(0).to(device)
    for idx,i in zip(range(len(x_unbatch)),claidx):
        if idx>=len(edge_index_unbatch):
            break
        H_emb = hpgnn.H_init[i]
        scores = torch.matmul(hg[idx], H_emb.T)
        att = F.softmax(scores, dim=-1)
        H_emb = torch.matmul(att, H_emb)
        edge_index_hat, sij = hpgnn.fg_model.get_explaination(x_unbatch[idx], H_emb, edge_index_unbatch[idx], edge_type_unbatch[idx], data.s[idx], data.t[idx], unbatched_data[idx].hyper_edge)
        sij_all = torch.cat([sij_all, sij])
    sij_all = 1-sij_all
    pred = hpgnn.forward(data.x, data.edge_index, data.edge_type, data.batch, data.links, data.s, data.t, data.hyper_edge)[0]
    pred_mask = hpgnn.forward(data.x, data.edge_index, data.edge_type, data.batch, data.links, data.s, data.t, data.hyper_edge, edge_weight=sij_all)[0]
    y = torch.argmax(data.y, dim=1)
    pred = pred[range(y.shape[0]), y]
    pred_mask = pred_mask[range(y.shape[0]), y]
    fidelity = torch.mean(pred - pred_mask)
        
    return fidelity

def eval_one_epoch(dataset_name, hpgnn, data_loader):
    if dataset_name == 'aug_citation' or dataset_name == 'synthetic':
        hpgnn.eval()
        # pbar = tqdm(data_loader)
        pbar = data_loader
        accuracy = []
        accuracy1 = []
        accuracy2 = []
        expacc = []
        jaccard = []
        roc_auc_all = []
        for idx, data in enumerate(pbar):
            predict_y, z, gemb, _, _, _, _ = hpgnn.forward(data.x, data.edge_index, data.edge_type, data.batch, data.links, data.s, data.t, data.hyper_edge, data.edge_gt_att)
            rauc, ja = explain_pg(predict_y, z, data, hpgnn, gemb)
            jaccard.append(ja)
            expacc.append(rauc)
            acc,acc1,acc2 = hpgnn.acc(predict_y, data.y)
            roc_auc = [roc_auc_score(data.y[:, i].cpu(), predict_y[:, i].cpu()) for i in range(data.y.shape[1]) if data.y[:, i].any()]
            roc_auc = sum(roc_auc)/len(roc_auc)

            accuracy.append(acc)
            accuracy1.append(acc1)
            accuracy2.append(acc2)
            roc_auc_all.append(roc_auc)

        print('roc_auc:{:.4f}'.format(sum(roc_auc_all)/len(roc_auc_all)))
        expacc = list(filter(None,expacc))
        jaccard = list(filter(None,jaccard))
        return sum(accuracy)/len(accuracy),sum(accuracy1)/len(accuracy1),sum(accuracy2)/len(accuracy2), sum(expacc)/len(expacc), sum(jaccard)/len(jaccard)
    elif dataset_name=='french_royalty' or dataset_name=='european_royalty':
        hpgnn.eval()
        # pbar = tqdm(data_loader)
        pbar = data_loader
        accuracy = []
        accuracy1 = []
        accuracy2 = []
        jaccard = []
        precision = []
        recall = []
        f1_score = []
        roc_auc_all = []
        for idx, data in enumerate(pbar):
            # data=data.to(device)
            predict_y, z, gemb, _,_, _, _ = hpgnn.forward(data.x, data.edge_index, data.edge_type, data.batch, data.links, data.s, data.t, data.hyper_edge, data.edge_gt_att)
            ja, p, r, f1 = explain_fr(predict_y, z, data, hpgnn, gemb)
            jaccard.append(ja)
            precision.append(p)
            recall.append(r)
            f1_score.append(f1)
            acc,acc1,acc2 = hpgnn.acc(predict_y, data.y)

            roc_auc = [roc_auc_score(data.y[:, i].cpu(), predict_y[:, i].cpu()) for i in range(data.y.shape[1]) if data.y[:, i].any()]
            roc_auc = sum(roc_auc)/len(roc_auc)

            accuracy.append(acc)
            accuracy1.append(acc1)
            accuracy2.append(acc2)
            roc_auc_all.append(roc_auc)

        print('roc_auc:{:.4f}'.format(sum(roc_auc_all)/len(roc_auc_all)))
        return sum(accuracy)/len(accuracy),sum(jaccard)/len(jaccard),sum(precision)/len(precision),sum(recall)/len(recall),sum(f1_score)/len(f1_score),sum(accuracy1)/len(accuracy1),sum(accuracy2)/len(accuracy2)
    elif dataset_name=='WN18RR' or dataset_name=='Amazon':
        hpgnn.eval()
        # pbar = tqdm(data_loader)
        pbar = data_loader
        accuracy = []
        accuracy1 = []
        accuracy2 = []
        fidelity = []
        roc_auc_all = []
        for idx, data in enumerate(pbar):
            data=data.to(device)
            predict_y, z, gemb, _,_, _, _ = hpgnn.forward(data.x, data.edge_index, data.edge_type, data.batch, data.links, data.s, data.t, data.hyper_edge)
            fi = explain_w(predict_y, z, data, hpgnn, gemb)
            fidelity.append(fi)
            acc,acc1,acc2 = hpgnn.acc(predict_y, data.y)

            roc_auc = [roc_auc_score(data.y[:, i].cpu(), predict_y[:, i].cpu()) for i in range(data.y.shape[1]) if data.y[:, i].any()]
            roc_auc = sum(roc_auc)/len(roc_auc)

            accuracy.append(acc)
            accuracy1.append(acc1)
            accuracy2.append(acc2)
            roc_auc_all.append(roc_auc)
            data = data.to('cpu')
        print('roc_auc:{:.4f}'.format(sum(roc_auc_all)/len(roc_auc_all)))
        return sum(accuracy)/len(accuracy),sum(accuracy1)/len(accuracy1),sum(accuracy2)/len(accuracy2), sum(fidelity)/len(fidelity)

def save_code(src_dir, dest_dir):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    for item in os.listdir(src_dir):
        s = os.path.join(src_dir, item)
        if os.path.isfile(s) and s.endswith('.py'):
            shutil.copy2(s, dest_dir)
    model_dir = os.path.join(src_dir, 'model')
    if os.path.exists(model_dir) and os.path.isdir(model_dir):
        dest_model_dir = os.path.join(dest_dir, 'model')
        if os.path.exists(dest_model_dir):
            shutil.rmtree(dest_model_dir)
        shutil.copytree(model_dir, dest_model_dir)

def train(dataset_name, hpgnn, dataloader, device, split_rate, epochs, lr, wd, alpha, H_init, class_weight, dataloader_test, dataset, save_path):
    best_acc = 0.0
    if dataset_name == 'aug_citation' or dataset_name == 'synthetic':
        best_r = 0.0
    elif dataset_name=='french_royalty' or dataset_name=='european_royalty':
        best_f1 = 0.0
        best_jaccard = 0.0
    elif dataset_name=='WN18RR' or dataset_name=='Amazon':
        best_fidelity = 0.0

    optimizer = torch.optim.Adam(list(hpgnn.parameters()), lr=lr)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss, acc, yloss, rloss, acc1, acc2, infoloss, pathloss, l2loss = train_one_epoch(hpgnn, dataloader, optimizer, device, split_rate, alpha, H_init, epoch, class_weight)
        print('Epoch {}, training loss: {:.4f}, training acc: {:.4f} {:.4f} {:.4f}; yloss: {:.4f}, rloss: {:.4f}, infoloss: {:.4f}, pathloss: {:.4f}, l2loss: {:.4f}'.format(epoch, loss, acc, acc1, acc2, yloss, rloss, infoloss, pathloss, l2loss))
        if epoch%10==9:
            with torch.no_grad():
                if dataset_name == 'aug_citation' or dataset_name == 'synthetic':
                    project_one_epoch(hpgnn, dataset, device)
                    acc, acc1, acc2, rocauc, jaccard = eval_one_epoch(dataset_name, hpgnn, dataloader_test)
                    print('TEST, test acc: {:.4f} {:.4f} {:.4f}, rocauc: {:.4f}, jaccard: {:.4f}'.format(acc, acc1, acc2, rocauc, jaccard))
                    if acc>best_acc or jaccard>best_r:
                        torch.save(hpgnn.state_dict(), save_path+"/hpgnn_{}.pkl".format(epoch))
                        best_acc =  acc if acc>best_acc else best_acc
                        best_r =  jaccard if jaccard>best_r else best_r
                    # project_one_epoch(hpgnn, dataset, device)
                elif dataset_name=='french_royalty' or dataset_name=='european_royalty':
                    project_one_epoch(hpgnn, dataset, device)
                    acc, jaccard, precision, recall, f1, acc1, acc2 = eval_one_epoch(dataset_name, hpgnn, dataloader_test)
                    print('TEST, test acc: {:.4f} {:.4f} {:.4f}, precision: {:.4f}, recall: {:.4f}, f1-score: {:.4f}, jaccard: {:.4f}'.format(acc, acc1, acc2, precision, recall, f1, jaccard))
                    if acc>best_acc or f1>best_f1 or jaccard>best_jaccard:
                        torch.save(hpgnn.state_dict(), save_path+"/hpgnn_{}.pkl".format(epoch))
                        best_acc =  acc if acc>best_acc else best_acc
                        best_f1 =  f1 if f1>best_f1 else best_f1
                        best_jaccard =  jaccard if jaccard>best_jaccard else best_jaccard
                elif dataset_name=='WN18RR' or dataset_name=='Amazon':
                    acc, acc1, acc2, fidelity = eval_one_epoch(dataset_name, hpgnn, dataloader_test)
                    print('TEST, test acc: {:.4f} {:.4f} {:.4f}, fidelity: {:.4f}'.format(acc, acc1, acc2, fidelity))
                    if acc>best_acc or fidelity>best_fidelity:
                        torch.save(hpgnn.state_dict(), save_path+"/hpgnn_{}.pkl".format(epoch))
                        best_acc =  acc if acc>best_acc else best_acc
                        best_fidelity =  fidelity if fidelity>best_fidelity else best_fidelity
                    project_one_epoch(hpgnn, dataset, device)

def test(dataset_name, hpgnn, dataloader):
    with torch.no_grad():
        if dataset_name == 'aug_citation' or dataset_name == 'synthetic':
            acc, acc1, acc2, rocauc, jaccard = eval_one_epoch(dataset_name, hpgnn, dataloader)
            print('TEST, test acc: {:.4f} {:.4f} {:.4f}, rocauc: {:.4f}, jaccard: {:.4f}'.format(acc, acc1, acc2, rocauc, jaccard))
        elif dataset_name=='french_royalty' or dataset_name=='european_royalty':
            acc, jaccard, precision, recall, f1, acc1, acc2 = eval_one_epoch(dataset_name, hpgnn, dataloader)
            print('TEST, test acc: {:.4f} {:.4f} {:.4f}, precision: {:.4f}, recall: {:.4f}, f1-score: {:.4f}, jaccard: {:.4f}'.format(acc, acc1, acc2, precision, recall, f1, jaccard))
        elif dataset_name=='WN18RR' or dataset_name=='Amazon':
            acc, acc1, acc2, fidelity = eval_one_epoch(dataset_name, hpgnn, dataloader)
            print('TEST, test acc: {:.4f} {:.4f} {:.4f}, fidelity: {:.4f}'.format(acc, acc1, acc2, fidelity))

def get_weight(labels):
    weight = torch.zeros(labels.max()+1).to(device)
    for i in range(labels.max()+1):
        weight[i] = labels.shape[0]/(labels==i).sum()
    return weight/(labels.max()+1)


def load_dataloader(dataset_name):
    if dataset_name=='french_royalty':
        # lr3=0.0001
        # alpha=1.5
        # proj=150
        # batch=64
        # neg=0.16
        # PE=200
        dataset = load_data_fr("./datasets/FrenchRoyalty/", dataset_name, 'full_data', random_state, splits)
        print(dataset['rel2idx'])
        A, links, labels, traces, weights = build_graph_fr(dataset['ent2idx'], dataset['rel2idx'], dataset['X_train'], dataset['X_train_triples'], negative_ratio, dataset['X_train_traces'], dataset['X_train_weights'])
        class_weight = get_weight(labels)
        mydataset = MyDataset_fr('./datasets/FrenchRoyalty/train_dataset_cpu7', A, links, labels, k, labels.max()+1, traces, weights, device, dataset['ent2idx']['UNK_ENT'])
        A_test, links_test, labels_test, traces_test, weights_test = build_graph_fr(dataset['ent2idx'], dataset['rel2idx'], dataset['X_test'], dataset['X_test_triples'], negative_ratio, dataset['X_test_traces'], dataset['X_test_weights'])
        mydataset_test = MyDataset_fr('./datasets/FrenchRoyalty/test_dataset_cpu7', A_test, links_test, labels_test, k, labels.max()+1, traces_test, weights_test, device, dataset['ent2idx']['UNK_ENT'])
        dataloader = DataLoader(mydataset, batch_size, shuffle=True, drop_last=False)
        dataloader_test = DataLoader(mydataset_test, batch_size, shuffle=True, drop_last=False)
        save_path = "./save_model/FrenchRoyalty/hpgnn7"
    elif dataset_name=='european_royalty':
        # lr3=0.0001
        # alpha=1.5
        # proj=150
        # batch=64
        # neg=0.25
        # PE=200
        dataset = load_data_fr("./datasets/EuropeanRoyalty/", dataset_name, 'test_full', random_state, splits)
        A, links, labels, traces, weights = build_graph_fr(dataset['ent2idx'], dataset['rel2idx'], dataset['X_train'], dataset['X_train_triples'], negative_ratio, dataset['X_train_traces'], dataset['X_train_weights'])
        class_weight = get_weight(labels)
        mydataset = MyDataset_fr('./datasets/EuropeanRoyalty/train_dataset_cpu2', A, links, labels, k, labels.max()+1, traces, weights, device, dataset['ent2idx']['UNK_ENT'])
        A_test, links_test, labels_test, traces_test, weights_test = build_graph_fr(dataset['ent2idx'], dataset['rel2idx'], dataset['X_test'], dataset['X_test_triples'], negative_ratio, dataset['X_test_traces'], dataset['X_test_weights'])
        mydataset_test = MyDataset_fr('./datasets/EuropeanRoyalty/test_dataset_cpu2', A_test, links_test, labels_test, k, labels.max()+1, traces_test, weights_test, device, dataset['ent2idx']['UNK_ENT'])
        dataloader = DataLoader(mydataset, batch_size, shuffle=True, drop_last=False)
        dataloader_test = DataLoader(mydataset_test, batch_size, shuffle=True, drop_last=False)
        save_path = "./save_model/EuropeanRoyalty/hpgnn2"
    elif dataset_name=='synthetic':
        # lr3=0.0005
        # alpha=2
        # proj=all
        # batch=128
        # PE=200
        load_dataset_pg('./datasets/Synthetic', dataset_name, 0.0, 0.2)
        g, train_link, test_link, exp, exp_path = load_dataset_pg1('./datasets/Synthetic', dataset_name)
        labels = train_link[1]
        num_class = int(labels.max().item()+1)
        class_weight = get_weight(labels)
        mydataset = MyDataset_pg(f'./datasets/Synthetic/{dataset_name}_train_dataset_cpu2.pth', g, train_link, k, 15, num_class, exp, exp_path, device, k_core, 5)
        dataloader = DataLoader(mydataset, batch_size, shuffle=True, drop_last=False)
        mydataset_test = MyDataset_pg(f'./datasets/Synthetic/{dataset_name}_test_dataset_cpu2.pth', g, test_link, k, 15, num_class, exp, exp_path, device, k_core, 5)
        dataloader_test = DataLoader(mydataset_test, batch_size, shuffle=True, drop_last=False)
        save_path = "./save_model/Synthetic/hpgnn1"  
    elif dataset_name=='aug_citation':
        # load_dataset_pg('./datasets/Aug-Citation', dataset_name, 0.0, 0.2)
        g, train_link, test_link, exp, exp_path = load_dataset_pg1('./datasets/Aug-Citation', dataset_name)
        labels = train_link[1]
        num_class = int(labels.max().item()+1)
        class_weight = get_weight(labels)
        mydataset = MyDataset_pg(f'./datasets/Aug-Citation/{dataset_name}_train_dataset_cpu4.pth', g, train_link, k, 30, num_class, exp, exp_path, device, k_core, 1) #aug30, syn15
        dataloader = DataLoader(mydataset, batch_size, shuffle=True, drop_last=False)
        mydataset_test = MyDataset_pg(f'./datasets/Aug-Citation/{dataset_name}_test_dataset_cpu4.pth', g, test_link, k, 30, num_class, exp, exp_path, device, k_core, 1)
        dataloader_test = DataLoader(mydataset_test, batch_size, shuffle=True, drop_last=False)
        save_path = "./save_model/Aug-Citation/hpgnn1"
    elif dataset_name=='WN18RR':
        # lr3=0.0001
        # alpha=1.5
        # proj=150
        # batch=32
        # neg=0.091
        # PE=200
        dataset = load_data_w("./datasets/WN18RR/")
        A, links, labels = build_graph_w(dataset['ent2idx'], dataset['rel2idx'], dataset['X_train'], dataset['X_train_triples'], negative_ratio)      
        class_weight = get_weight(labels)
        A_test, links_test, labels_test = build_graph_w(dataset['ent2idx'], dataset['rel2idx'], dataset['X_train'], dataset['X_test_triples'], negative_ratio)
        mydataset = MyDataset_w('./datasets/WN18RR/train_dataset_cpu11', A, links, labels, k, labels.max()+1, device)
        mydataset_test = MyDataset_w('./datasets/WN18RR/test_dataset_cpu11', A_test, links_test, labels_test, k, labels.max()+1, device)
        dataloader = DataLoader(mydataset, batch_size, shuffle=True, drop_last=False)
        dataloader_test = DataLoader(mydataset_test, batch_size, shuffle=True, drop_last=False)
        save_path = "./save_model/WN18RR/hpgnn9"
    elif dataset_name=='Amazon':
        dataset = load_data_a("./datasets/Amazon/")
        A, links, labels = build_graph_w(dataset['ent2idx'], dataset['rel2idx'], dataset['X_train'], dataset['X_train_triples'], negative_ratio)      
        class_weight = get_weight(labels)
        A_test, links_test, labels_test = build_graph_w(dataset['ent2idx'], dataset['rel2idx'], dataset['X_train'], dataset['X_test_triples'], negative_ratio)
        mydataset = MyDataset_w('./datasets/Amazon/train_dataset_cpu4', A, links, labels, k, labels.max()+1, device)
        mydataset_test = MyDataset_w('./datasets/Amazon/test_dataset_cpu4', A_test, links_test, labels_test, k, labels.max()+1, device)
        dataloader = DataLoader(mydataset, batch_size, shuffle=True, drop_last=False)
        dataloader_test = DataLoader(mydataset_test, batch_size, shuffle=True, drop_last=False)
        save_path = "./save_model/Amazon/hpgnn1"
    return dataloader, dataloader_test, labels.max()+1, class_weight, mydataset, save_path

def main():
    dataloader, dataloader_test, pred_num, class_weight, mydataset, save_path = load_dataloader(dataset_name)
    # save_code("../HPGNN-Link/",save_path+"/src/")
    fe_model = RGCN(k+1, nhid, pred_num, num_bases, lr1, dropout, weight_decay, grad_norm, device).to(device)
    fg_model = Generator(nfeat=nhid, nhid=nhid, max_num_nodes=max_num_nodes, dropout=dropout, lr=lr2, weight_decay=weight_decay, device=device).to(device)
    
    H_init = torch.rand((pred_num,K,nhid))

    hpgnn = HPGNN(fe_model, fg_model, nhid, device, pred_num, K, H_init.clone(), max_length).to(device)

    hpgnn.load_state_dict(torch.load("./save_model/FrenchRoyalty/hpgnn5/hpgnn_419.pkl",  map_location=device))
    # hpgnn.fg_model.load_state_dict(pretrained_parameters)
    test(dataset_name, hpgnn, dataloader_test)
    train(dataset_name, hpgnn, dataloader, device, split_ratio, epochs, lr, weight_decay, alpha, H_init, class_weight, dataloader_test, mydataset, save_path)
    test(dataset_name, hpgnn, dataloader_test)



if __name__ == '__main__':
    main()