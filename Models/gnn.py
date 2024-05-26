import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
import torch.nn.functional as F
from torch.nn import Linear, ReLU, ModuleList, Parameter
from torch_geometric.nn.inits import uniform

from Models.conv import GNN_node, GNN_node_Virtualnode

from torch_scatter import scatter_mean

class GNN(torch.nn.Module):

    def __init__(self, num_classes, num_tasks, num_layer = 5, emb_dim = 300, 
                    gnn_type = 'gin', virtual_node = True, residual = False, drop_ratio = 0.5, JK = "last", graph_pooling = "mean",
                    node_encoder = lambda x: x, edge_encoder = lambda x: x, use_node_encoder = True, graph_features = 0, num_mlp_layers = 1, dim_pooling = False,
                 node_broadcast = False):
        '''
            num_tasks (int): number of labels to be predicted
            virtual_node (bool): whether to add virtual node or not
        '''

        super(GNN, self).__init__()
        
        print("Old GNN implementation.")

        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks
        self.graph_pooling = graph_pooling
        self.num_classes = num_classes
        self.num_tasks = num_tasks
        self.use_node_encoder = use_node_encoder
        self.graph_features = graph_features
        self.num_mlp_layers = num_mlp_layers
        self.dim_pooling = dim_pooling
        self.node_broadcast = node_broadcast

        if self.num_layer < 1:
            raise ValueError("Number of GNN layers must be at least 1.")

        ### GNN to generate node embeddings
        if virtual_node:
            self.gnn_node = GNN_node_Virtualnode(num_layer, emb_dim, JK = JK, drop_ratio = drop_ratio, residual = residual, gnn_type = gnn_type, node_encoder=node_encoder, edge_encoder=edge_encoder)
        else:
            self.gnn_node = GNN_node(num_layer, emb_dim, JK = JK, drop_ratio = drop_ratio, residual = residual, gnn_type = gnn_type, node_encoder=node_encoder, edge_encoder=edge_encoder)

        self.graph_emb_dim = emb_dim if (JK != "concat") else (emb_dim*(num_layer+1))
        self.set_mlp(graph_features)

        ### Pooling function to generate whole-graph embeddings
        print(f"graph_pooling: {graph_pooling}")
        if self.graph_pooling == "sum":
            self.pool = global_add_pool
        elif self.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif self.graph_pooling == "max":
            self.pool = global_max_pool
        elif self.graph_pooling == "attention":
            self.pool = GlobalAttention(gate_nn = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, 1)))

        # Dimensional pooling (for CWN)
        if self.dim_pooling:
            print("CWN: Using dimensional pooling.")
            self.lin_per_dim = torch.nn.ModuleList()
            self.relu = torch.nn.ReLU()
            for _ in range(3):
                self.lin_per_dim.append(torch.nn.Linear(self.graph_emb_dim, self.graph_emb_dim))
        
    def forward(self, batched_data):
        # Concatenate global features with node features
        if self.node_broadcast:
            if self.graph_features > 0:
                global_features = self.standardize_features(batched_data.graph_features[:, :self.graph_features])
                # Repeat the global features for each node in the batch
                global_features_repeated = global_features[batched_data.batch]

                batched_data.x = torch.cat([batched_data.x, global_features_repeated], dim=1)
        else:
            h_node = self.gnn_node(batched_data)

            if self.dim_pooling:
                dimensional_pooling = []
                for dim in range(3):
                    multiplier = torch.unsqueeze(batched_data.node_type[:, dim], dim=1)
                    single_dim = h_node * multiplier
                    single_dim = self.pool(single_dim, batched_data.batch)
                    single_dim = self.relu(self.lin_per_dim[dim](single_dim))
                    dimensional_pooling.append(single_dim)
                h_graph = sum(dimensional_pooling)
            else:
                h_graph = self.pool(h_node, batched_data.batch)

            # Attach graph level features
            # if self.graph_features > 0:
            #     h_graph = torch.cat([h_graph,  batched_data.graph_features[:, 0:self.graph_features]], dim=1)

            if self.graph_features > 0 and (not self.node_broadcast):
                h_graph = torch.cat([h_graph,  self.standardize_features(batched_data.graph_features[:, 0:self.graph_features])], dim=1)

            h_graph = h_graph
            for i, layer in enumerate(self.mlp):
                h_graph = layer(h_graph)
                if i < self.num_mlp_layers - 1:
                    h_graph = F.dropout(h_graph, self.drop_ratio, training = self.training)

            if self.num_tasks == 1:
                h_graph = h_graph.view(-1, self.num_classes)
            else:
                h_graph.view(-1, self.num_tasks, self.num_classes)
            return h_graph
        h_node = self.gnn_node(batched_data)

        if self.dim_pooling:
            dimensional_pooling = []
            for dim in range(3):
                multiplier = torch.unsqueeze(batched_data.node_type[:, dim], dim=1)
                single_dim = h_node * multiplier
                single_dim = self.pool(single_dim, batched_data.batch)
                single_dim = self.relu(self.lin_per_dim[dim](single_dim))
                dimensional_pooling.append(single_dim)
            h_graph = sum(dimensional_pooling)
        else:
            h_graph = self.pool(h_node, batched_data.batch)

        h_graph = h_graph
        for i, layer in enumerate(self.mlp):
            h_graph = layer(h_graph)
            if i < self.num_mlp_layers - 1:
                h_graph = F.dropout(h_graph, self.drop_ratio, training=self.training)

        if self.num_tasks == 1:
            h_graph = h_graph.view(-1, self.num_classes)
        else:
            h_graph.view(-1, self.num_tasks, self.num_classes)
        return h_graph

    def freeze_gnn(self, freeze = True):
        # Frezze GNN layers to allow us to tune the MLP separately
        for param in self.gnn_node.parameters():
            param.requires_grad = False

    def set_mlp(self, graph_features = 0, copy_emb_weights = False):
        self.graph_features = graph_features
        hidden_size = self.graph_emb_dim // 2
        new_mlp = ModuleList([])
        new_mlp.requires_grad = False

        
        for i in range(self.num_mlp_layers):
            if (self.node_broadcast):
                in_size = hidden_size if i > 0 else self.graph_emb_dim
            else:
                in_size = hidden_size if i > 0 else self.graph_emb_dim + graph_features
            out_size = hidden_size if i < self.num_mlp_layers - 1 else self.num_classes*self.num_tasks

            new_linear_layer = Linear(in_size, out_size)

            if copy_emb_weights:
                copy_len = self.emb_dim if i == 0 else hidden_size

                new_linear_layer.weight.requires_grad = False
                new_linear_layer.weight[:, 0:copy_len] = self.mlp[2*i].weight[:, 0:copy_len].detach().clone()
                new_linear_layer.weight.requires_grad = True

                new_linear_layer.bias.requires_grad = False
                new_linear_layer.bias = Parameter(self.mlp[2*i].bias.detach().clone())
                new_linear_layer.bias.requires_grad = True

            new_mlp.append(new_linear_layer)

            if self.num_mlp_layers > 0 and i < self.num_mlp_layers - 1:
                new_mlp.append(ReLU())

        new_mlp.requires_grad = True
        self.mlp = new_mlp

    def standardize_features(self, features, max_value=1000):
        min_val = features.min(dim=0, keepdim=True)[0]
        max_val = features.max(dim=0, keepdim=True)[0]

        # Scale to [0, 1]
        standardized_features = (features - min_val) / (max_val - min_val)

        # Scale to [0, max_value]
        standardized_features = standardized_features * max_value

        # Convert to integers
        standardized_features = standardized_features.round().long()

        return standardized_features

if __name__ == '__main__':
    GNN(num_tasks = 10)