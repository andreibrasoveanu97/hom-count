import os
import csv

import torch
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import ZINC, GNNBenchmarkDataset, TUDataset, QM9
import torch.optim as optim
from torch_geometric.transforms import ToUndirected, Compose, OneHotDegree
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from ogb.utils.features import get_atom_feature_dims
from Models.gin import GIN
from Models.gnn import GNN
from Models.encoder import NodeEncoder, EdgeEncoder
from Models.mlp import MLP
from Misc.attach_graph_features import AttachGraphFeat
from Misc.drop_features import DropFeatures
from Misc.add_zero_edge_attr import AddZeroEdgeAttr
from Misc.pad_node_attr import PadNodeAttr
from datasets.PeptidesStructural import PeptidesStructuralDataset
from datasets.PeptidesFunctional import PeptidesFunctionalDataset
from torch_geometric.data import Dataset
import numpy as np

implemented_TU_datasets = ["mutag", "proteins", "nci1", "nci109"]
shuffled_datasets = ["mutag", "proteins", "nci1", "nci109", "qm9"]
class CustomSubset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

def get_transform(args, split = None):
    transforms = []
    if args.dataset.lower() == "csl":
        transforms.append(OneHotDegree(5))
        
    # Pad features if necessary (needs to be done after adding additional features from other transformation)
    if args.dataset.lower() == "csl":
        transforms.append(AddZeroEdgeAttr(args.emb_dim))
        transforms.append(PadNodeAttr(args.emb_dim))

    if args.graph_feat != "":
        use_zinc = args.dataset.lower() == "zinc"
        use_shuffled = args.dataset.lower() in shuffled_datasets
        transforms.append(AttachGraphFeat(args.graph_feat,
                                          process_splits_separately = use_zinc,
                                          half_nr_edges = use_zinc,
                                          misaligned = args.use_misaligned,
                                          shuffle=use_shuffled,
                                          dummy=args.dummy))
        
    if args.do_drop_feat:
        transforms.append(DropFeatures(args.emb_dim))

    return Compose(transforms)

def load_dataset(args, config):
    transform = get_transform(args)

    if transform is None:
        dir = os.path.join(config.DATA_PATH, args.dataset, "Original")
    else:
        print(repr(transform))
        trafo_str = repr(transform).replace("\n", "")
        dir = os.path.join(config.DATA_PATH, args.dataset, trafo_str)

    if args.dataset.lower() == "zinc":
        datasets = [ZINC(root=dir, subset=True, split=split, pre_transform=transform) for split in ["train", "val", "test"]]
    elif args.dataset.lower() == "cifar10":
        datasets = [GNNBenchmarkDataset(name ="CIFAR10", root=dir, split=split, pre_transform=Compose([ToUndirected(), transform])) for split in ["train", "val", "test"]]
    elif args.dataset.lower() == "cluster":
        datasets = [GNNBenchmarkDataset(name ="CLUSTER", root=dir, split=split, pre_transform=transform) for split in ["train", "val", "test"]]
    elif args.dataset.lower() in ["ogbg-molhiv", "ogbg-ppa", "ogbg-code2", "ogbg-molpcba", "ogbg-moltox21", "ogbg-molesol", "ogbg-molbace", "ogbg-molbbbp", "ogbg-molclintox", "ogbg-molmuv", "ogbg-molsider", "ogbg-moltoxcast", "ogbg-molfreesolv", "ogbg-mollipo"]:
        dataset = PygGraphPropPredDataset(root=dir, name=args.dataset.lower(), pre_transform=transform)
        split_idx = dataset.get_idx_split()
        datasets = [dataset[split_idx["train"]], dataset[split_idx["valid"]], dataset[split_idx["test"]]]
    elif args.dataset.lower() == "csl":
        all_idx = {}
        for section in ['train', 'val', 'test']:
            with open(os.path.join(config.SPLITS_PATH, "CSL",  f"{section}.index"), 'r') as f:
                reader = csv.reader(f)
                all_idx[section] = [list(map(int, idx)) for idx in reader]
        dataset = GNNBenchmarkDataset(name ="CSL", root=dir, pre_transform=transform)
        datasets = [dataset[all_idx["train"][args.split]], dataset[all_idx["val"][args.split]], dataset[all_idx["test"][args.split]]]
    elif args.dataset.lower() in ["exp", "cexp"]:
        dataset = PlanarSATPairsDataset(name=args.dataset, root=dir, pre_transform=transform)
        split_dict = dataset.separate_data(args.seed, args.split)
        datasets = [split_dict["train"], split_dict["valid"], split_dict["test"]]
    elif args.dataset.lower() in implemented_TU_datasets:
        dataset = TUDataset(root=dir, name=args.dataset, pre_transform=transform,
                                                      use_node_attr=False, use_edge_attr=False)
        indices_train = lambda: [i for i, data in enumerate(dataset) if data.split == 'train']
        indices_test = lambda: [i for i, data in enumerate(dataset) if data.split == 'test']
        indices_valid = lambda: [i for i, data in enumerate(dataset) if data.split == 'valid']

        datasets = [CustomSubset(dataset, indices_train()), CustomSubset(dataset, indices_valid()), CustomSubset(dataset, indices_test())]
    elif args.dataset.lower() == "qm9":
        dataset = QM9(root = dir, pre_transform=transform)

        indices_train = lambda: [i for i, data in enumerate(dataset) if data.split == 'train']
        indices_test = lambda: [i for i, data in enumerate(dataset) if data.split == 'test']
        indices_valid = lambda: [i for i, data in enumerate(dataset) if data.split == 'valid']

        train_split = standardize_y(CustomSubset(dataset, indices_train()))
        valid_split = standardize_y(CustomSubset(dataset, indices_valid()))
        test_split = standardize_y(CustomSubset(dataset, indices_test()))

        datasets = [train_split, valid_split, test_split]
    elif args.dataset.lower() == "peptides":
        dataset = PeptidesStructuralDataset(root = dir, pre_transform=transform)
        splits = dataset.get_idx_split()

        datasets = [dataset[splits["train"]], dataset[splits["val"]], dataset[splits["test"]]]
    elif args.dataset.lower() == "peptides-functional":
        dataset = PeptidesFunctionalDataset(root=dir, pre_transform=transform)
        splits = dataset.get_idx_split()

        datasets = [dataset[splits["train"]], dataset[splits["val"]], dataset[splits["test"]]]
    else:
        raise NotImplementedError("Unknown dataset")

    train_loader = DataLoader(datasets[0], batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(datasets[1], batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(datasets[2], batch_size=args.batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

def get_model(args, num_classes, num_vertex_features, num_tasks):
    node_feature_dims = []
    
    model = args.model.lower()

    if args.dataset.lower() == "zinc" and not args.do_drop_feat:
        node_feature_dims.append(21)
        node_encoder = NodeEncoder(emb_dim=args.emb_dim, feature_dims=node_feature_dims)
        edge_encoder = EdgeEncoder(emb_dim=args.emb_dim, feature_dims=[4])
    elif args.dataset.lower() in ["ogbg-molhiv", "ogbg-molpcba", "ogbg-moltox21", "ogbg-molesol", "ogbg-molbace", "ogbg-molbbbp", "ogbg-molclintox", "ogbg-molmuv", "ogbg-molsider", "ogbg-moltoxcast", "ogbg-molfreesolv", "ogbg-mollipo"] and not args.do_drop_feat:

        node_feature_dims += get_atom_feature_dims()
        print("node_feature_dims: ", node_feature_dims)
        node_encoder, edge_encoder = NodeEncoder(args.emb_dim, feature_dims=node_feature_dims), EdgeEncoder(args.emb_dim)
    elif args.dataset.lower() == "mutag":
        for i in range (0, 7):
            node_feature_dims.append(2)
        node_encoder, edge_encoder = NodeEncoder(args.emb_dim, feature_dims=node_feature_dims), EdgeEncoder(args.emb_dim, feature_dims=[2, 2, 2, 2])
    elif args.dataset.lower() == "qm9":
        node_encoder, edge_encoder = NodeEncoder(args.emb_dim, feature_dims=[2, 2, 2, 2, 2, 10, 1, 1, 1, 1, 5]), EdgeEncoder(args.emb_dim, feature_dims=[2, 2, 2, 2])
    elif args.dataset.lower() in ["peptides", "peptides-functional"]:
        node_encoder, edge_encoder = NodeEncoder(args.emb_dim, feature_dims=[17, 3, 7, 7, 5, 1, 6, 2, 2]), EdgeEncoder(args.emb_dim, feature_dims=[4, 1, 2])
    elif args.dataset.lower() in ["proteins"]:
        node_encoder, edge_encoder = NodeEncoder(args.emb_dim, feature_dims=[2, 2, 2]), EdgeEncoder(args.emb_dim, feature_dims=[2])
    else:
        node_encoder, edge_encoder = lambda x: x, lambda x: x

    max_struct_size = args.max_struct_size
    if max_struct_size > 0 and not args.cliques:
        max_struct_size = 2

    if model in ["gin", "gcn", "gat"]:
        return GNN(num_classes, num_tasks, args.num_layers, args.emb_dim, 
                gnn_type = model, virtual_node = args.use_virtual_node, drop_ratio = args.drop_out, JK = "last", 
                graph_pooling = args.pooling, edge_encoder=edge_encoder, node_encoder=node_encoder, 
                use_node_encoder = args.use_node_encoder, num_mlp_layers = args.num_mlp_layers)
    elif model == "mlp":
            return MLP(num_features=num_vertex_features, num_layers=args.num_layers, hidden=args.emb_dim, 
                    num_classes=num_classes, num_tasks=num_tasks, dropout_rate=args.drop_out, graph_pooling=args.pooling)
    elif model == "gin_tu":
        return GIN(num_features=num_vertex_features, num_layers=args.num_layers, hidden=args.emb_dim, num_classes=num_classes,
                   dropout_rate=args.drop_out, max_cell_dim=max_struct_size, dimensional_pooling=args.use_explicit_pattern_enc,
                   readout=args.pooling, num_mlp_layers=args.num_mlp_layers)
    else: # Probably don't need other models
        pass

    return model


def get_optimizer_scheduler(model, args, finetune = False):
    
    if finetune:
        lr = args.lr2
    else:
        lr = args.lr
    optimizer = optim.Adam(model.parameters(), lr=lr)

    if args.lr_scheduler == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                                    args.lr_scheduler_decay_steps,
                                                    gamma=args.lr_scheduler_decay_rate)
    elif args.lr_scheduler == 'None':
        scheduler = None
    elif args.lr_scheduler == "ReduceLROnPlateau":
         scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                                mode='min',
                                                                factor=args.lr_scheduler_decay_rate,
                                                                patience=args.lr_schedule_patience,
                                                                verbose=True)
    else:
        raise NotImplementedError(f'Scheduler {args.lr_scheduler} is not currently supported.')

    return optimizer, scheduler

def get_loss(args):
    metric_method = None
    if args.dataset.lower() in ["zinc", "qm9", "peptides"]:
        loss = torch.nn.L1Loss()
        metric = "mae"
    elif args.dataset.lower() in ["ogbg-molesol", "ogbg-molfreesolv", "ogbg-mollipo"]:
        loss = torch.nn.L1Loss()
        metric = "rmse (ogb)"
        metric_method = get_evaluator(args.dataset)
    elif args.dataset.lower() in ["cifar10", "csl", "exp", "cexp"]:
        loss = torch.nn.CrossEntropyLoss()
        metric = "accuracy"
    elif args.dataset in ["ogbg-molhiv", "ogbg-moltox21", "ogbg-molbace", "ogbg-molbbbp", "ogbg-molclintox", "ogbg-molsider", "ogbg-moltoxcast"]:
        loss = torch.nn.BCEWithLogitsLoss()
        metric = "rocauc (ogb)" 
        metric_method = get_evaluator(args.dataset)
    elif args.dataset == "ogbg-ppa":
        loss = torch.nn.BCEWithLogitsLoss()
        metric = "accuracy (ogb)" 
        metric_method = get_evaluator(args.dataset)
    elif args.dataset in ["ogbg-molpcba", "ogbg-molmuv"]:
        loss = torch.nn.BCEWithLogitsLoss()
        metric = "ap (ogb)" 
        metric_method = get_evaluator(args.dataset)
    elif args.dataset.lower() in ["proteins", "mutag"]:
        loss = torch.nn.BCEWithLogitsLoss()
        metric = "accuracy"
    elif args.dataset.lower() in ["peptides-functional"]:
        loss = torch.nn.BCEWithLogitsLoss()
        metric = "ap"
    else:
        raise NotImplementedError("No loss for this dataset")
    
    return {"loss": loss, "metric": metric, "metric_method": metric_method}

def get_evaluator(dataset):
    evaluator = Evaluator(dataset)
    eval_method = lambda y_true, y_pred: evaluator.eval({"y_true": y_true, "y_pred": y_pred})
    return eval_method


def standardize_y(dataset):
    num_targets = dataset[0].y.shape[1]

    for i in range(num_targets):
        mean = torch.mean(dataset[:].data.y[:, i])
        std = torch.std(dataset[:].data.y[:, i])

        dataset[:].data.y[:, i] = (dataset[:].data.y[:, i] - mean) / std

    return dataset