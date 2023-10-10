import sys

import networkx as nx
from torch_geometric.datasets import TUDataset, QM9
from torch_geometric.transforms import Compose, OneHotDegree
from torch_geometric import utils
from ogb.nodeproppred import PygNodePropPredDataset
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
import numpy as np
from Misc.add_zero_edge_attr import AddZeroEdgeAttr
from Misc.pad_node_attr import PadNodeAttr
from Misc.drop_features import DropFeatures
import argparse
import json
import grinpy as gp
implemented_TU_datasets = ["mutag", "proteins", "nci1", "nci109"]

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
    parser.add_argument("dataset", help="Choose the dataset for which the global features are computed")
    parser.add_argument("feature",
                        help="Choose which global feature you want to compute: [wiener, hosoya, circuit_rank, second_eigen, spectral_radius]")
    args = parser.parse_args()

    transform = get_transform(args)

    counts_dict = {}

    # only one feature per counts.json file
    counts_dict["pattern_sizes"] = [1]

    counts_dict["data"] = []

    if args.dataset.lower() in implemented_TU_datasets:
        dataset = TUDataset(root="./datasets/", name=args.dataset, pre_transform=transform,
                            use_node_attr=True, use_edge_attr=False)
        dataset.shuffle()
        datasets = [dataset[:int(len(dataset) * 0.8)],
                    dataset[int(len(dataset) * 0.8):int(len(dataset) * 0.9)],
                    dataset[int(len(dataset) * 0.9):]]
    elif args.dataset.lower() == "qm9":
        dataset = QM9(root="./datasets/")
        dataset.shuffle()
        datasets = [dataset[:int(len(dataset) * 0.8)],
                    dataset[int(len(dataset) * 0.8):int(len(dataset) * 0.9)],
                    dataset[int(len(dataset) * 0.9):]]
    else:
        raise NotImplementedError("Unknown dataset")

    splits = ["train", "valid", "test"]

    if (args.dataset.lower() in implemented_TU_datasets) or (args.dataset.lower() == "qm9"):
        idx = 0
        for split, dataset in zip(splits, datasets):
            for i, graph in enumerate(dataset):
                g = utils.to_networkx(graph)

                counts_dict["data"].append({"vertices": graph.num_nodes,
                                            "edges": graph.num_edges,
                                            "split": split,
                                            "idx_in_split": i,
                                            "idx": idx,
                                            "counts": [float(compute_feature(feature=args.feature, graph=g))]})
                idx += 1


        counts_dict["data"] = sorted(counts_dict["data"], key=lambda x: x["idx"])
    else:
        raise NotImplementedError("Preloaded global features are not implemented for this dataset")

    with open('./Counts/GlobalFeatures/{}_{}_global.json'.format(args.dataset.upper(), args.feature.upper()), 'w') as fp:
        json.dump(counts_dict, fp)

# sum of the lengths of the shortest paths between all pairs of vertices
def wiener_index(graph):
    return -1 if nx.wiener_index(graph) == float("inf") else nx.wiener_index(graph)

def hosoya_index(graph):
    return len(nx.max_weight_matching(graph.to_undirected(), maxcardinality = True))

def eigenvalues_laplacian(graph):
    return nx.laplacian_spectrum(graph.to_undirected())

# sum of squares of the degrees of the vertices
def zagreb_index1(graph):
    return sum(map(lambda x: graph.degree(x)**2, graph.nodes))

# sum of the products of the degrees of pairs of adjacent vertices
def zagreb_index2(graph):
    return sum(map(lambda edge: graph.degree(edge[0]) * graph.degree(edge[1]), graph.edges))

def circuit_rank(graph):
    g = graph.to_undirected()
    return (g.number_of_edges() - g.number_of_nodes() + nx.number_connected_components(g))

def spectral_radius(graph):
    return max(map(abs, nx.adjacency_spectrum(graph)))

def diameter(graph):
    return nx.diameter(graph)

def independence_no(graph):
    return int(gp.independence_number(graph))

def dummy(graph):
    return np.float64(1)

def compute_feature(feature, graph):
    match feature:
        case "wiener":
            return wiener_index(graph)
        case "hosoya":
            return hosoya_index(graph)
        case "circuit_rank":
            return circuit_rank(graph)
        case "second_eigen":
            return eigenvalues_laplacian(graph)[1]
        case "spectral_radius":
            return spectral_radius(graph)
        case "zagreb_m1":
            return zagreb_index1(graph)
        case "zagreb_m2":
            return zagreb_index2(graph)
        case "diameter":
            return diameter(graph)
        case "independence":
            return independence_no(graph)
        case "dummy":
            return dummy(graph)
        case "zagreb_m22":
            return zagreb_index2(graph)
if __name__ == "__main__":
    main()