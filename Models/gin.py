import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, BatchNorm1d as BN
from torch_geometric.nn import GINConv, JumpingKnowledge
import torch.nn as nn
from torch.nn import Linear, ReLU, Sequential
from torch_geometric.nn import GINConv, JumpingKnowledge, global_mean_pool

from Models.utils import get_nonlinearity, get_pooling_fn
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder

class GIN(torch.nn.Module):
    def __init__(self, num_features, num_layers, hidden, num_classes, max_cell_dim = 0,mode='cat', readout='sum', num_tasks = 1,
                 dropout_rate=0.5, nonlinearity='relu', dimensional_pooling = True):
        super(GIN, self).__init__()
        self.max_cell_dim = max_cell_dim
        self.pooling_fn = get_pooling_fn(readout)
        self.dropout_rate = dropout_rate
        self.num_tasks = num_tasks
        self.num_classes = num_classes
        self.nonlinearity = nonlinearity
        conv_nonlinearity = get_nonlinearity(nonlinearity, return_module=True)
        self.conv1 = GINConv(
            Sequential(
                Linear(num_features, hidden),
                BN(hidden),
                conv_nonlinearity(),
                Linear(hidden, hidden),
                BN(hidden),
                conv_nonlinearity(),
            ), train_eps=False)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(
                GINConv(
                    Sequential(
                        Linear(hidden, hidden),
                        BN(hidden),
                        conv_nonlinearity(),
                        Linear(hidden, hidden),
                        BN(hidden),
                        conv_nonlinearity(),
                    ), train_eps=False))


        self.dimensional_pooling = dimensional_pooling
        if max_cell_dim > 0 and self.dimensional_pooling:
            self.lin_per_dim = nn.ModuleList()
            for i in range(max_cell_dim + 1):
                if mode == 'cat':
                    self.lin_per_dim.append(Linear(num_layers * hidden, hidden))
                else:
                    self.lin_per_dim.append(Linear(hidden, hidden))
                self.lin_per_dim.append(Linear(num_layers * hidden, hidden))
        else:
            if mode == 'cat':
                self.lin1 = Linear(num_layers * hidden, hidden)
            else:
                self.lin1 = Linear(hidden, hidden)

        self.jump = JumpingKnowledge(mode)
        self.lin2 = Linear(hidden, num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.jump.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        model_nonlinearity = get_nonlinearity(self.nonlinearity, return_module=False)
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x.float(), edge_index)
        xs = [x]
        for conv in self.convs:
            x = conv(x, edge_index)
            xs += [x]
        x = self.jump(xs)

        if self.max_cell_dim > 0 and self.dimensional_pooling:
            dimensional_pooling = []
            for dim in range(self.max_cell_dim + 1):
                multiplier = torch.unsqueeze(data.x[:, dim], dim=1)
                single_dim = x * multiplier
                single_dim = self.pooling_fn(single_dim, batch)
                single_dim = model_nonlinearity(self.lin_per_dim[dim](single_dim))
                dimensional_pooling.append(single_dim)
            x = sum(dimensional_pooling)
        else:
            x = self.pooling_fn(x, batch)
            x = model_nonlinearity(self.lin1(x))

        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.lin2(x)
        return x

    def __repr__(self):
        return self.__class__.__name__
