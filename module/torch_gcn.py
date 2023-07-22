import math
from torch.nn.modules.module import Module
import torch
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import numpy as np


class GraphConvolution(Module):
    def __init__(self, in_features, out_features,bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):            
        support = torch.mm(input, self.weight) 
        output = torch.spmm(adj, support)      
        if self.bias is not None:
            return output + self.bias          
        else:
            return output                      

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GCN(nn.Module):
    def __init__(self,  nfea=256, nclass=256,dropout = 0.1):
        super(GCN, self).__init__()

        self.temp1 = 128
        self.temp2 = 256
        self.temp3 = 512

        self.dropout = dropout

        self.gc1 = GraphConvolution(nfea, self.temp2)
        self.fc3 = nn.Linear(self.temp2,nclass)



    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout)
        x = F.dropout(x, self.dropout)
        x = F.softmax(self.fc3(x), dim=0)
        return x       




class SAGE(Module):
    def __init__(self,in_features=256, out_features=256,dropout=0.1) -> None:
        super().__init__()
        self.gnn = GCN(in_features,out_features)
        self.fc = nn.Linear(out_features,out_features)
        self.dropout = dropout
    
    def forward(self,x,adj):
        x = F.relu(self.gnn(x,adj))
        x = F.dropout(x,self.dropout)
        x = F.relu(self.fc(x))
        x = F.dropout(x,self.dropout)
        return x

