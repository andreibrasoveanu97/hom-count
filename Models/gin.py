import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, BatchNorm1d as BN
from torch_geometric.nn import GINConv, JumpingKnowledge
import torch.nn as nn
from torch.nn import Linear, ReLU, Sequential, ModuleList, Parameter
from torch_geometric.nn import GINConv, JumpingKnowledge, global_mean_pool

from Models.utils import get_nonlinearity, get_pooling_fn
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder

class GIN(torch.nn.Module):
    def __init__(self, num_features, num_layers, hidden, num_classes, max_cell_dim = 0,mode='cat', readout='sum', num_tasks = 1,
                 dropout_rate=0.5, nonlinearity='relu', dimensional_pooling = True, num_mlp_layers = 1):
        super(GIN, self).__init__()
        self.max_cell_dim = max_cell_dim
        self.pooling_fn = get_pooling_fn(readout)
        self.dropout_rate = dropout_rate
        self.num_tasks = num_tasks
        self.emb_dim = hidden
        self.num_mlp_layers = num_mlp_layers
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

    def set_mlp(self, graph_features=0, copy_emb_weights=False):
        self.graph_features = graph_features
        hidden_size = self.emb_dim // 2
        new_mlp = ModuleList([])
        new_mlp.requires_grad = False

        for i in range(self.num_mlp_layers):
            in_size = hidden_size if i > 0 else self.emb_dim + graph_features
            out_size = hidden_size if i < self.num_mlp_layers - 1 else self.num_classes * self.num_tasks

            new_linear_layer = Linear(in_size, out_size)

            if copy_emb_weights:
                copy_len = self.emb_dim if i == 0 else hidden_size

                new_linear_layer.weight.requires_grad = False
                new_linear_layer.weight[:, 0:copy_len] = self.mlp[2 * i].weight[:, 0:copy_len].detach().clone()
                new_linear_layer.weight.requires_grad = True

                new_linear_layer.bias.requires_grad = False
                new_linear_layer.bias = Parameter(self.mlp[2 * i].bias.detach().clone())
                new_linear_layer.bias.requires_grad = True

            new_mlp.append(new_linear_layer)

            if self.num_mlp_layers > 0 and i < self.num_mlp_layers - 1:
                new_mlp.append(ReLU())

        new_mlp.requires_grad = True
        self.mlp = new_mlp
    def __repr__(self):
        return self.__class__.__name__
