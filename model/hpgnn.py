import torch.nn as nn
import torch
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj

M = 6
T = 1
Q = 100

def hook_y(grad):
    print(grad)



class HPGNN(nn.Module):

    def __init__(self, fe_model, fg_model, embedding_dim, device, num_class, K, H_init, max_length=3):
        super().__init__()
        self.fe_model = fe_model
        self.fg_model = fg_model
        self.device = device
        self.embedding_dim = embedding_dim
        self.num_class = num_class
        self.K = K
        self.H_init = nn.Parameter(H_init)
        self.y_init = torch.repeat_interleave(torch.eye(num_class),dim=0,repeats=K).to(device)

        self.lin1 = nn.Linear(embedding_dim, embedding_dim ,bias=False)
        self.lin2 = nn.Linear(embedding_dim, embedding_dim ,bias=False)
        self.last_layer = nn.Linear(H_init.shape[0]*H_init.shape[1], H_init.shape[0]*H_init.shape[1],bias=False)
        self.last_pool = nn.MaxPool1d(H_init.shape[1])
        self.cal_path = False
        self.max_length = max_length
        self.path_weight = nn.Parameter(torch.ones(self.max_length)*0.5)
        self.epsilon = torch.tensor(1e-6)
        
    def activate_path(self, x):
        self.cal_path = x

    def forward(self, x, edge_index, edge_type, batch, links, s, t, hyper_edge, edge_gt_att=None, edge_weight=None):
        if edge_weight is not None:
            hn = self.fe_model.forward(x, edge_index, edge_type, edge_weight)
            hg = self.fe_model.graph_forward(x, edge_index, edge_type, batch, links, edge_weight)
        else:
            hn = self.fe_model.forward(x, edge_index, edge_type)
            hg = self.fe_model.graph_forward(x, edge_index, edge_type, batch, links)
        sim_new = []
        dist_new = []
        L_info = 0
        L_path = 0
        L_l2 = 0
        if self.training and self.cal_path:
            Adj = to_dense_adj(edge_index, batch).detach()
        for i in range(self.num_class):
            H_emb = self.H_init[i]
            scores = torch.matmul(hg, H_emb.T)
            att = F.softmax(scores, dim=-1)
            H_emb = torch.matmul(att, H_emb)

            edge_index_hat, edge_type_hat, sij_hat = self.fg_model.forward(hn, H_emb, edge_index, edge_type, batch, hyper_edge)
            if edge_weight is not None:
                hg_pro = self.fe_model.graph_forward(x, edge_index_hat, edge_type_hat, batch, links, sij_hat*edge_weight)
            else:
                hg_pro = self.fe_model.graph_forward(x, edge_index_hat, edge_type_hat, batch, links, sij_hat)
            sim = torch.sum(torch.matmul(hg_pro, H_emb.T)/(torch.norm(hg_pro, p=2, dim=-1, keepdim=True)*torch.norm(H_emb, p=2, dim=-1, keepdim=True)),dim=1)
            L_info += self.infoloss(sij_hat)
            dist_new.append(1/scores.unsqueeze(0))
            sim_new.append(sim.unsqueeze(0))
            if self.training and self.cal_path:
                path_tmp, l2_tmp = self.path_loss3(edge_index_hat, sij_hat, batch, Adj, s, t)
                L_path += path_tmp
                L_l2 += l2_tmp

        sim_new = torch.cat(sim_new,dim=0)
        logits = sim_new.T
        pred = F.softmax(logits, dim=1)
        return pred, hn, hg, torch.cat(dist_new,dim=0), L_info/(self.num_class), L_path/(self.num_class), L_l2/(self.num_class)

    def l2_norm(self, x):
        return x/(torch.max(torch.norm(x, dim=1, keepdim=True), self.epsilon))
    
    def project_forward(self, x, edge_index, edge_type, batch, links, H, hyper_edge):
        hn = self.fe_model.forward(x, edge_index, edge_type)
        H_emb = H.unsqueeze(0)[torch.zeros(batch.max()+1,dtype=torch.int64)]
        edge_index_hat, edge_type_hat, sij_hat = self.fg_model.forward(hn, H_emb, edge_index, edge_type, batch, hyper_edge)
        hg_pro = self.fe_model.graph_forward(x, edge_index_hat, edge_type_hat, batch, links, sij_hat)                 
        dist = torch.sum(torch.square(hg_pro-H_emb),dim=1)
        sim = torch.log((dist+1)/(dist+1e-4))
        return hg_pro[sim.argmax()], sim.max()
    
    def infoloss(self, att):
        r=0.5
        info_loss = (att * torch.log(att/r + 1e-6) + (1-att) * torch.log((1-att)/(1-r+1e-6) + 1e-6)).mean()
        return info_loss
    
    def path_loss3(self, edge_index, sij, batch, Adj, s, t):
        Mdj = to_dense_adj(edge_index, batch, sij.squeeze(1))
        weight = torch.clamp(self.path_weight, min = 1e-10, max = 1.0)
        l2_loss = torch.sum(sij.pow(2))/2
        u = Mdj[torch.arange(Mdj.shape[0]), s, :]
        mk = Mdj[torch.arange(Mdj.shape[0]), :, t]

        a = Adj[torch.arange(Adj.shape[0]), s, :]
        ak = Adj[torch.arange(Adj.shape[0]), :, t]

        uk = torch.pow(Mdj[torch.arange(Mdj.shape[0]), s, t]/check_zero(Adj[torch.arange(Adj.shape[0]), s, t]),1)
        
        if self.max_length==2:
            u2 = Mdj[torch.arange(Mdj.shape[0]), t, :]
            mk2 = Mdj[torch.arange(Mdj.shape[0]), :, s]
            a2 = Adj[torch.arange(Adj.shape[0]), t, :]
            ak2 = Adj[torch.arange(Adj.shape[0]), :, s]
            uk += torch.pow(Mdj[torch.arange(Mdj.shape[0]), t, s]/check_zero(Adj[torch.arange(Adj.shape[0]), t, s]),1)

        for l in range(self.max_length-1):
            utmp = u * mk
            atmp = a * ak
            tmp = utmp/check_zero(atmp)
            if self.max_length==2:
                utmp2 = u2 * mk2
                atmp2 = a2 * ak2
                utmp3 = u * u2
                atmp3 = a * a2
                utmp4 = mk * mk2
                atmp4 = ak * ak2
                tmp = torch.cat([tmp, utmp2/check_zero(atmp2),
                                utmp3/check_zero(atmp3), utmp4/check_zero(atmp4)],dim=1)
            w = torch.pow(tmp.max(dim=1)[0]+1e-8,1/(l+1))
            uk += weight[l+1] * w
            if self.max_length>2:
                u = torch.bmm(u.unsqueeze(1), Mdj).squeeze(1)
                a = torch.bmm(a.unsqueeze(1), Adj).squeeze(1)
        uk = uk / (self.max_length)
        loss = -torch.mean(torch.log(uk+1e-8))
        return loss, l2_loss/Mdj.shape[0]
    
    def acc(self, pred, y):
        if torch.isnan(pred).any():
            print('a')
        num1 = torch.sum(pred.argmax(dim=1)==y.argmax(dim=1))
        num = torch.sum(pred[pred.argmax(dim=1)==y.argmax(dim=1)].max(dim=1)[0]>0.5)
        idx = torch.where(y.argmax(dim=1)!=0)
        num2 = torch.sum(pred[idx].argmax(dim=1)==y[idx].argmax(dim=1))
        return (num/y.shape[0]),(num1/y.shape[0]),(num2/idx[0].shape[0])
    
    def BCEacc(self, pred, y):
        res = {}
        num = {}
        for i in range(self.num_class):
            tmp_pred = pred[y.argmax(dim=-1)==i][:,i]
            res[i] = (tmp_pred>0.5).sum()
            num[i] = len(tmp_pred)
        return res,num
    
    def cls_loss(self, sim, y):
        L_clst = self.clst_loss(sim,y)
        L_sep = self.sep_loss(sim, y)
        L_div = self.div_loss()
        L_cls = L_clst+0.1*L_sep + 0.001*L_div
        return L_cls
    
    def c_loss(self, pred, y, class_weight):
        lossfunc = torch.nn.CrossEntropyLoss(weight=class_weight)
        L_c  = lossfunc(pred, y)
        return L_c
    
    def clst_loss(self, sim, y):
        tmp = sim.transpose(0,1)[y.bool()]
        L_clst = torch.mean(tmp.max(dim=1)[0])
        return L_clst
    
    def sep_loss(self, sim, y):
        tmp = sim.transpose(0,1)[~y.bool()].reshape(sim.shape[1],-1)
        L_sep = 1/torch.mean(tmp.mean(dim=1))
        return L_sep
    
    def div_loss(self):
        L_div = 0
        for i in range(self.num_class):
            proto = self.H_init[i]
            proto = F.normalize(proto, p=2, dim=1)
            matrix1 = torch.mm(proto, torch.t(proto)) - torch.eye(proto.shape[0]).to(self.device)
            matrix2 = torch.zeros(matrix1.shape).to(self.device)
            L_div += torch.sum(torch.where(matrix1 > 0, matrix1, matrix2))
        return L_div
    
def check_zero(x):
    return torch.where(x==0, x+1e-8, x)