import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from .layer import RGCNConv
from tqdm import tqdm

def hook_y(grad):
    print(grad)

class RGCN(torch.nn.Module):
    def __init__(self, nfeat, nhid, nrel_class, num_bases, lr=0.001, dropout=0.5, weight_decay=5e-4, grad_norm=1, device=None):
        super(RGCN, self).__init__()

        self.nrel_class = nrel_class     
        
        self.latent_dim=2
        self.convs = torch.nn.ModuleList()
        self.convs.append(RGCNConv(nfeat, nhid, nrel_class, num_bases=num_bases))
        for i in range(0, self.latent_dim-1):
            self.convs.append(RGCNConv(nhid, nhid, nrel_class, num_bases=num_bases))

        self.pool = global_mean_pool
        self.lr = lr
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.device = device
        self.grad_norm = grad_norm
        self.fc_out = nn.Sequential(nn.Linear(nhid, nrel_class))
    
    def forward(self, x, edge_index, edge_type, edge_weight=None):
        for i,conv in zip(range(self.latent_dim), self.convs):
            x = conv(x, edge_index, edge_type, edge_weight)
            x = torch.sigmoid(x)
            x = F.dropout(x, p = self.dropout, training = self.training)
        return x
    
    def graph_forward(self, x, edge_index, edge_type, batch, links, edge_weight=None):
        for i,conv in zip(range(self.latent_dim), self.convs):
            x = conv(x, edge_index, edge_type, edge_weight)
            x = torch.sigmoid(x)
            x = F.dropout(x, p = self.dropout, training = self.training)

        graph_emb = self.pool(x, batch = batch)
        graph_emb = x[(links==1).squeeze(1)] * graph_emb * x[(links==2).squeeze(1)]
        return graph_emb
    
    def get_gemb(self, gemb):
        gemb = F.relu(self.lin(gemb))
        gemb = F.dropout(gemb, p=self.dropout, training=self.training)
        return gemb

    def distmult(self, gemb):
        pred = self.fc_out(gemb)
        pred = F.softmax(pred,dim=-1)
        return pred


    def c_loss(self, pred, y, class_weight):
        lossfunc = torch.nn.CrossEntropyLoss(weight=class_weight)
        L_c  = lossfunc(pred, y)
        return L_c
    
    def acc(self, pred, y):
        num = torch.sum(pred.argmax(dim=1)==y.argmax(dim=1))
        return (num/y.shape[0])
    
    def initialize(self):
        for i in range(0, self.latent_dim):
            self.convs[i].reset_parameters()

    def train_one_epoch(self, dataloader, optimizer, class_weight):
        self.train()
        pbar = dataloader
        train_loss = []
        accuracy = []
        for idx, data in enumerate(pbar):
            data = data.to(self.device)
            gemb = self.graph_forward(data.x, data.edge_index, data.edge_type, data.batch, data.links)
            predict_y = self.distmult(gemb)
            loss = self.c_loss(predict_y, data.y, class_weight)
            train_loss.append(loss)
            acc = self.acc(predict_y, data.y)
            accuracy.append(acc)
            loss.backward()
            optimizer.step()
        return sum(train_loss)/len(train_loss), sum(accuracy)/len(accuracy)

    def eval_one_epoch(self, data_loader):
        self.eval()
        # pbar = tqdm(data_loader)
        pbar = data_loader
        accuracy = []
        for idx, data in enumerate(pbar):
            data = data.to(self.device)
            gemb = self.graph_forward(data.x, data.edge_index, data.edge_type, data.batch, data.links)
            predict_y = self.distmult(gemb)
            acc = self.acc(predict_y, data.y)
            accuracy.append(acc)    
        return sum(accuracy)/len(accuracy)

    def train_gcn(self, loaders, epochs, dataloader_test, class_weight, initialize=True):
        if initialize:
            self.initialize()

        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        for epoch in range(epochs):
            optimizer.zero_grad()
            loss,acc = self.train_one_epoch(loaders, optimizer, class_weight)
            if epoch%10==0:
                acc_test = self.eval_one_epoch(dataloader_test)
                print('Epoch {}, training loss: {:.4f}, training acc: {:.4f}, test acc: {:.4f}'.format(epoch, loss, acc, acc_test))

