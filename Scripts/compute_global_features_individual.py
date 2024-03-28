import networkx as nx
from torch_geometric.datasets import ZINC, TUDataset, QM9
from torch_geometric.transforms import Compose, OneHotDegree
from torch_geometric import utils
from ogb.graphproppred import PygGraphPropPredDataset
import numpy as np
from Misc.add_zero_edge_attr import AddZeroEdgeAttr
from Misc.pad_node_attr import PadNodeAttr
# from PeptidesStructural import PeptidesStructuralDataset
import argparse
import json
import grinpy as gp
implemented_TU_datasets = ["mutag", "proteins", "nci1", "nci109", "qm9"]

def to_sagegraph(graph):
    G = Graph()
    from_networkx_graph(G, graph.to_undirected())
    return G

def get_hosoya(graph):
    return sum(map(abs, matching_polynomial(graph).coefficients()))

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
    parser.add_argument("feature", help = "Choose which global feature you want to compute: [wiener, hosoya, circuit_rank, second_eigen, spectral_radius]")
    args = parser.parse_args()

    transform = get_transform(args)

    if (args.dataset.lower() == "zinc"):
        datasets = [ZINC(root="./datasets/", subset=True, split=split, pre_transform=transform) for split in
                    ["train", "val", "test"]]
    elif args.dataset.lower() in ["ogbg-molhiv", "ogbg-ppa", "ogbg-code2", "ogbg-molpcba", "ogbg-moltox21", "ogbg-molesol", "ogbg-molbace", "ogbg-molbbbp", "ogbg-molclintox", "ogbg-molmuv", "ogbg-molsider", "ogbg-moltoxcast", "ogbg-molfreesolv", "ogbg-mollipo"]:
        dataset = PygGraphPropPredDataset(root="./datasets/", name=args.dataset.lower(), pre_transform=transform)
        split_idx = dataset.get_idx_split()
        datasets = [dataset[split_idx["train"]], dataset[split_idx["valid"]], dataset[split_idx["test"]]]
    elif args.dataset.lower() in implemented_TU_datasets:
        dataset = TUDataset(root='./datasets/', name=args.dataset,
                            use_node_attr=True, use_edge_attr=False)
        dataset.shuffle()
        datasets = [dataset[:int(len(dataset) * 0.8)],
                    dataset[int(len(dataset) * 0.8):int(len(dataset) * 0.9)],
                    dataset[int(len(dataset) * 0.9):]]
    elif args.dataset.lower() == "qm9":
        dataset = QM9(root='./datasets/')
        dataset.shuffle()
        datasets = [dataset[:int(len(dataset) * 0.8)],
                    dataset[int(len(dataset) * 0.8):int(len(dataset) * 0.9)],
                    dataset[int(len(dataset) * 0.9):]]
    elif args.dataset.lower() == "peptides":
        dataset = PeptidesStructuralDataset(root="./datasets/")
        splits = dataset.get_idx_split()

        datasets = [dataset[splits["train"]], dataset[splits["val"]], dataset[splits["test"]]]
    else:
        raise NotImplementedError("Unknown dataset")

    counts_dict = {}

    # only one feature per counts.json file
    if args.feature != "all":
        counts_dict["pattern_sizes"] = [1]
    else:
        counts_dict["pattern_sizes"] = [1, 2, 3, 4, 5, 6, 7, 8]

    counts_dict["data"] = []

    splits = ["train", "valid", "test"]
    if (args.dataset.lower() == "zinc"):
        idx = 0
        for split, dataset in zip(splits, datasets):
            for i, graph in enumerate(dataset):
                g = utils.to_networkx(graph)

                counts_dict["data"].append({"vertices": graph.num_nodes,
                                            "edges": graph.num_edges / 2,
                                            "split": split,
                                            "idx_in_split": i,
                                            "idx": idx,
                                            "counts": [float(compute_feature(feature=args.feature, graph=g))]})
                idx += 1
    elif args.dataset.lower() in ["ogbg-molhiv", "ogbg-ppa", "ogbg-code2", "ogbg-molpcba", "ogbg-moltox21", "ogbg-molesol", "ogbg-molbace", "ogbg-molbbbp", "ogbg-molclintox", "ogbg-molmuv", "ogbg-molsider", "ogbg-moltoxcast", "ogbg-molfreesolv", "ogbg-mollipo"]:
        for split, dataset in zip(splits, datasets):
            for i, graph in enumerate(dataset):
                g = utils.to_networkx(graph)

                counts_dict["data"].append({"vertices": graph.num_nodes,
                                            "edges": graph.num_edges,
                                            "split": split,
                                            "idx_in_split": i,
                                            "idx": int(split_idx[split][i]),
                                            "counts": [float(compute_feature(feature=args.feature, graph=g))]})

        counts_dict["data"] = sorted(counts_dict["data"], key=lambda x: x["idx"])
    elif args.dataset.lower() in ["peptides"]:
        splits_dict = dataset.get_idx_split()
        idx = 0
        splits = ["train", "val", "test"]
        for split, dataset in zip(splits, datasets):
            for i, graph in enumerate(dataset):
                g = utils.to_networkx(graph)

                if args.feature != "all":
                    counts_dict["data"].append({"vertices": graph.num_nodes,
                                            "edges": graph.num_edges,
                                            "split": split,
                                            "idx_in_split": i,
                                            "idx": int(splits_dict[split][i]),
                                            "counts": [float(compute_feature(feature=args.feature, graph=g))]})
                else:
                    counts_dict["data"].append({"vertices": graph.num_nodes,
                                                "edges": graph.num_edges,
                                                "split": split,
                                                "idx_in_split": i,
                                                "idx": int(splits_dict[split][i]),
                                                "counts": [float(wiener_index(g)),
                                                           float(hosoya_index(g)),
                                                           float(independence_no(g)),
                                                           float(eigenvalues_laplacian(g)[1]),
                                                           float(circuit_rank(g)),
                                                           float(spectral_radius(g)),
                                                           float(zagreb_index1(g)),
                                                           float(zagreb_index2(g))]})
                idx += 1
        counts_dict["data"] = sorted(counts_dict["data"], key=lambda x: x["idx"])
    else:
        NotImplementedError("Dataset not implemented")

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