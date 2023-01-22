import json

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform

class AttachGraphFeat(BaseTransform):
    r""" 
    """
    def __init__(self, path_graph_feat: str, process_splits_separately = False, half_nr_edges = False):
        self.path_graph_feat = path_graph_feat
        self.half_nr_edges = half_nr_edges
        
        with open(path_graph_feat, 'r') as file:
            graph_features = json.load(file)

            if type(graph_features) is dict:
                graph_features = graph_features["data"]   

        # Compute mean and standard deviation of training data for standardization
        training_counts = torch.stack(list(map(lambda f: torch.tensor(f['counts']) ,(filter(lambda f: f['split'] == 'train', graph_features)))))
        self.mean  = torch.mean(training_counts)
        self.std = torch.std(training_counts)

        if process_splits_separately:
            new_graph_feat = []
            for split in ["train", "val", "test"]:
                graph_features_split = list(filter(lambda g: g["split"] == split, graph_features))
                graph_features_split.sort(key = lambda f: f['idx_in_split'])
                new_graph_feat += graph_features_split

        for i, g in enumerate(graph_features):
            g['idx'] = i

        graph_features.sort(key = lambda f: f['idx'])

        self.idx = 0
        self.graph_features = graph_features


    def __call__(self, data: Data):
        assert  self.graph_features[self.idx]['vertices'] == data.x.shape[0]

        if self.half_nr_edges:
            assert  self.graph_features[self.idx]['edges']*2 == data.edge_index.shape[1]
        else:
            assert  self.graph_features[self.idx]['edges'] == data.edge_index.shape[1]

        # Standardize data via standard score (https://en.wikipedia.org/wiki/Standard_score)
        graph_features = (torch.tensor(self.graph_features[self.idx]['counts']) - self.mean) / self.std
        graph_features = torch.unsqueeze(graph_features, 0)
        data.graph_features = graph_features
        self.idx += 1
        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.path_graph_feat})'