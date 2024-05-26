import json

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
import random

import torch


def normalize_and_round(tensor, new_min=0, new_max=100):
    """
    Normalize the tensor to the range [new_min, new_max] and round to integers.
    """
    # Normalize the tensor to the range [0, 1]
    min_val = torch.min(tensor)
    max_val = torch.max(tensor)
    normalized = (tensor - min_val) / (max_val - min_val)

    # Scale to the new range and round to integers
    scaled = normalized * (new_max - new_min) + new_min
    integer_scaled = torch.round(scaled).int()

    return integer_scaled

class AttachGraphFeat(BaseTransform):
    r""" 
    """

    def __init__(self, path_graph_feat: str, process_splits_separately=False, half_nr_edges=False, misaligned=False,
                 shuffle=False, dummy=False, check_vertices=True, node_features = False):
        self.path_graph_feat = path_graph_feat
        self.half_nr_edges = half_nr_edges
        self.dummy = dummy
        self.shuffle = shuffle
        self.check_vertices = check_vertices
        self.node_features = node_features

        with open(path_graph_feat, 'r') as file:
            graph_features = json.load(file)

            if type(graph_features) is dict:
                graph_features = graph_features["data"]

        # for no predefined dataset train-test splits, shuffle them on the spot
        if shuffle:
            # keep idx_in_split and split together as they are both related to the split
            values_to_shuffle = [(d['idx_in_split'], d['split']) for d in graph_features]

            random.shuffle(values_to_shuffle)

            for i, d in enumerate(graph_features):
                d['idx_in_split'], d['split'] = values_to_shuffle[i]

        # Compute mean and standard deviation of training data for standardization
        training_counts = torch.stack(
            list(map(lambda f: torch.tensor(f['counts']), (filter(lambda f: f['split'] == 'train', graph_features)))))

        self.mean = torch.mean(training_counts, dim=0, keepdim=False)
        self.std = torch.std(training_counts, dim=0, keepdim=False)

        print(f"training_counts: {training_counts.shape}, mean: {self.mean.shape}, std: {self.std.shape}")
        print(self.mean)
        print(self.std)
        # Mask to mask out constant values
        self.mask = self.std != 0

        self.misaligned = misaligned

        if process_splits_separately:
            new_graph_feat = []
            for split in ["train", "val", "test"]:
                graph_features_split = list(filter(lambda g: g["split"] == split, graph_features))
                graph_features_split.sort(key=lambda f: f['idx_in_split'])
                new_graph_feat += graph_features_split

        for i, g in enumerate(graph_features):
            g['idx'] = i
        if not self.misaligned:
            graph_features.sort(key=lambda f: f['idx'])
        else:
            # Add features the wrong way
            graph_features.sort(key=lambda f: -f['idx'])

        self.idx = 0
        self.graph_features = graph_features

    def __call__(self, data: Data):
        # Only perform a sanity check for not misaligned features

        if not self.misaligned and self.check_vertices:
            assert self.graph_features[self.idx]['vertices'] == data.x.shape[0]

            if self.half_nr_edges:
                assert self.graph_features[self.idx]['edges'] * 2 == data.edge_index.shape[1]
            else:
                assert self.graph_features[self.idx]['edges'] == data.edge_index.shape[1]

        # Standardize data via standard score (https://en.wikipedia.org/wiki/Standard_score)
        splits = self.graph_features[self.idx]["split"]
        node_features = (torch.tensor(self.graph_features[self.idx]['counts']))

        if not self.dummy:
            # graph_features = (torch.tensor(self.graph_features[self.idx]['counts']) - self.mean) / self.std
            # graph_features = graph_features[self.mask]
            graph_features = torch.tensor(self.graph_features[self.idx]['counts'])
        else:
            graph_features = torch.tensor(self.graph_features[self.idx]['counts'])
        # Mask out values that were constant (they would be NaN after dividing by std)


        # Standardize by dividing by the maximal number that each pattern can appear in the graph

        # graph_features = torch.tensor(self.graph_features[self.idx]['counts'])
        # print(graph_features)
        # nr_vertices_vec = torch.ones_like(graph_features)*data.x.shape[0]
        # max_nr_counts = torch.pow(nr_vertices_vec, graph_features)
        # graph_features = graph_features / max_nr_counts

        # print(graph_features)
        # print("\n")
        # assert torch.all(torch.le(graph_features, 1)) and torch.all(torch.ge(graph_features, 0))

        graph_features = torch.unsqueeze(graph_features, 0)

        if not self.node_features:
            data.graph_features = graph_features
        else:
            node_features = node_features.unsqueeze(0).expand(data.x.shape[0], -1)
            data.x = torch.cat((data.x, node_features), dim=1)
        if self.shuffle:
            data.edge_attr = torch.unsqueeze(torch.tensor([1]), 0)
        data.split = splits

        self.idx += 1
        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.path_graph_feat}, {self.misaligned})'