import networkx as nx
import torch
from torch_geometric.datasets import ZINC, GNNBenchmarkDataset
from torch_geometric.transforms import ToUndirected, Compose, OneHotDegree
from torch_geometric import utils
from torch_geometric.loader import DataLoader
from ogb.nodeproppred import PygNodePropPredDataset
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from Misc.add_zero_edge_attr import AddZeroEdgeAttr
from Misc.pad_node_attr import PadNodeAttr
import argparse
import json

def get_transform(args, split=None):
    transforms = []
    if args.dataset.lower() == "csl":
        transforms.append(OneHotDegree(5))

    # Pad features if necessary (needs to be done after adding additional features from other transformation)
    if args.dataset.lower() == "csl":
        transforms.append(AddZeroEdgeAttr(args.emb_dim))
        transforms.append(PadNodeAttr(args.emb_dim))

    # if args.do_drop_feat:
    #     transforms.append(DropFeatures(args.emb_dim))

    return Compose(transforms)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help = "Choose the dataset for which the global features are computed")
    args = parser.parse_args()

    transform = get_transform(args)

    if (args.dataset.lower() == "zinc"):
        datasets = [ZINC(root="./Datasets/", subset=True, split=split, pre_transform=transform) for split in
                    ["train", "val", "test"]]
    else:
        raise NotImplementedError("Unknown dataset")

    counts_dict = {}

    # 1: wiener index
    # 2: hosoya index
    # 3, 4: first and second eigenvalue of the Laplacian
    counts_dict["pattern_sizes"] = [1, 2, 3, 4]
    counts_dict["data"] = []

    splits = ["train", "valid", "test"]

    for split, dataset in zip(splits, datasets):
        for i, graph in enumerate(dataset):
            g = utils.to_networkx(graph)
            wiener_idx = nx.wiener_index(g)
            hosoya_idx = len(nx.max_weight_matching(g.to_undirected(), maxcardinality=True))
            first_eigen = nx.laplacian_spectrum(g.to_undirected())[0]
            second_eigen = nx.laplacian_spectrum(g.to_undirected())[1]

            counts_dict["data"].append({"vertices": graph.num_nodes,
                                        "edges": graph.num_edges,
                                        "split": split,
                                        "idx_in_split": i,
                                        "counts": [wiener_idx, hosoya_idx, first_eigen, second_eigen]})

    with open('./Counts/GlobalFeatures/{}_full_global.json'.format(args.dataset.upper()), 'w') as fp:
        json.dump(counts_dict, fp)

if __name__ == "__main__":
    main()