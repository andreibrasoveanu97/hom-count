import networkx as nx
from torch_geometric.datasets import ZINC, TUDataset, QM9
from torch_geometric.transforms import Compose, OneHotDegree
from torch_geometric import utils
from ogb.graphproppred import PygGraphPropPredDataset
import numpy as np
from Misc.add_zero_edge_attr import AddZeroEdgeAttr
from Misc.pad_node_attr import PadNodeAttr
#from PeptidesStructural import PeptidesStructuralDataset
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

def homo_lumo_index(g):
    if not nx.is_connected(g.to_undirected()):
        return False

    n = g.number_of_nodes()

    spectrum = nx.adjacency_spectrum(g.to_undirected())

    if n % 2 == 0:
        h = int(n / 2 - 1)  # because array indices start from 0 instead of 1
        l = int(h + 1)
        return max([abs(spectrum[h]), abs(spectrum[l])])

    h = int((n - 1) / 2)
    return abs(spectrum[h])

def eccentricity(graph):
    """ Eccentricity of the graph for all its vertices"""
    return nx.floyd_warshall_numpy(graph).max(axis=0).tolist()

def eccentric_connectivity_index(graph):
    """ Eccentric Connectivity Index
    Graph must be connected, otherwise it cannot be computed"""
    degrees = [val for (node, val) in graph.degree()]
    return sum( map( lambda a,b: a*b, degrees, eccentricity(graph) ) )

def connectivity_index(graph, power):
    """Connectivity index (R)"""
    E = list(graph.edges())
    degrees = [val for (node, val) in graph.degree()]

    return np.float64(sum(map(lambda x: (degrees[x[0]] * degrees[x[1]]) ** power, E)))

def randic_index(graph):
    """Randic Index"""
    return connectivity_index(graph, -0.5)

def atom_bond_connectivity_index(graph):
    """ Atom-Bond Connectivity Index (ABC) """
    s = np.longdouble(0)  # summator
    E = list(graph.edges())
    degrees = [val for (node, val) in graph.degree()]

    for (u, v) in E:
        d1 = np.float64(degrees[u])
        d2 = np.float64(degrees[v])
        s += np.longdouble(((d1 + d2 - 2) / (d1 * d2)) ** .5)
    return np.float64(s)

def spectrum_matrix(graph, type = "adjacency"):
    match type:
        case "adjacency":
            return nx.adjacency_spectrum(graph.to_undirected())
        case "laplacian":
            return nx.laplacian_spectrum(graph.to_undirected())
        case "distance":
            dist_matrix = nx.floyd_warshall_numpy(graph)
            s = np.linalg.eigvalsh(dist_matrix).tolist()
            return s

def estrada_index(graph):
    """Estrada Index (EE)"""
    spectrum = nx.adjacency_spectrum(graph.to_undirected())

    return sum(map(lambda x: np.exp(x.real), spectrum))

def distance_estrada_index(graph):
    """Distance Estrada Index (DEE)"""
    spectrum = spectrum_matrix(graph, "distance")

    return sum(map(lambda x: np.exp(x.real), spectrum))

def degree_distance(graph):
    """Degree Distance (DD)"""
    all_pairs_shortest_paths = dict(nx.all_pairs_shortest_path_length(graph))
    degrees = [val for (node, val) in graph.degree()]

    degree_distance = 0
    for u in graph.nodes():
        for v in graph.nodes():
            if u != v:
                distance_uv = all_pairs_shortest_paths[u][v]
                degree_distance += (degrees[u] + degrees[v]) * distance_uv

    return degree_distance


def molecular_topological_index_mti(graph):
    A = nx.adjacency_matrix(graph).toarray()

    D = dict(nx.all_pairs_shortest_path_length(graph))
    D_matrix = np.zeros((len(graph), len(graph)))
    for i, d in D.items():
        for j, dist in d.items():
            D_matrix[i][j] = dist

    degrees = [graph.degree(i) for i in range(len(graph))]

    d = np.array(degrees)
    E = (A + D_matrix) @ d
    mti = np.sum(E)

    return mti

def eccentric_distance_sum_index(graph):
    return (eccentricity(graph)*nx.floyd_warshall_numpy(graph).sum(axis = 1)).sum()


def balaban_j_index(graph):
    n = len(graph.nodes())
    m = len(graph.edges())

    # Compute the number of connected components (c) and the circuit rank (gamma)
    c = nx.number_connected_components(graph.to_undirected())
    gamma = m - n + c
    # Compute the graph distance matrix
    D = nx.floyd_warshall_numpy(graph).sum(axis=1)

    sum_distances = 0.0

    for (i, j) in graph.to_undirected().edges():
        sum_distances += 1.0 / np.sqrt(D[i] * D[j])

    # Compute the Balaban index J
    J = (m / (gamma + 1.0)) * sum_distances
    return J

def balaban_j_index_cyclomatic(graph):
    n = len(graph.nodes())
    m = len(graph.edges())

    # Compute the number of connected components (c) and the circuit rank (gamma)
    c = nx.number_connected_components(graph.to_undirected())
    gamma = 1.0
    # Compute the graph distance matrix
    D = nx.floyd_warshall_numpy(graph).sum(axis=1)

    sum_distances = 0.0

    for (i, j) in graph.to_undirected().edges():
        sum_distances += 1.0 / np.sqrt(D[i] * D[j])

    # Compute the Balaban index J
    J = (m / (gamma + 1.0)) * sum_distances
    return J

def resistance_matrix(graph):
    n = graph.number_of_nodes()
    L = nx.laplacian_matrix(graph.to_undirected()).toarray()
    Gamma = np.linalg.inv(L + (np.identity(n) / n))

    Omega = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            Omega[i, j] = Gamma[i, i] + Gamma[j, j] - 2 * Gamma[i, j]
    return Omega

def kirchoff_index(graph):
    resis_mat = resistance_matrix(graph)
    return resis_mat.sum() / 2.0

def terminal_wiener_index(graph):
    s = 0
    n = graph.number_of_nodes()
    distances = nx.floyd_warshall_numpy(graph)
    degrees = [graph.degree(i)/2 for i in range(n)]

    for u in range(n):
        if degrees[u] != 1: continue
        for v in range(u+1, n):
            if degrees[v] == 1:
                s += distances[u, v]

    return s

def reverse_wiener_index(graph):
    wi = wiener_index(graph)
    diam = diameter(graph)
    n = graph.number_of_nodes()

    return diam * n * (n-1) / 2 - wi

def hyper_wiener_index(graph):
    distances = nx.floyd_warshall_numpy(graph)
    return (np.power(distances, 2).sum() + distances.sum())/4.0

def reciprocal_distance_matrix(graph):
    distances = nx.floyd_warshall_numpy(graph)

    reciprocal_mat = np.zeros_like(distances, dtype = float)

    for i in range(len(distances)):
        for j in range(len(distances[i])):
            if distances[i][j] != 0:
                reciprocal_mat[i][j] = 1 / distances[i][j]

    return reciprocal_mat

def harary_index(graph):
    reciprocal_mat = reciprocal_distance_matrix(graph)

    return reciprocal_mat.sum() / 2.0

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
        case "homo_lumo":
            return homo_lumo_index(graph)
        case "eccentric":
            return eccentric_connectivity_index(graph)
        case "randic":
            return randic_index(graph)
        case "abc":
            return atom_bond_connectivity_index(graph)
        case "estrada":
            return estrada_index(graph)
        case "estrada_distance":
            return distance_estrada_index(graph)
        case "dd":
            return degree_distance(graph)
        case "mti":
            return molecular_topological_index_mti(graph)
        case "eccentric_distance_sum":
            return eccentric_distance_sum_index(graph)
        case "balaban_j":
            return balaban_j_index(graph)
        case "balaban_cyclomatic":
            return balaban_j_index_cyclomatic(graph)
        case "kirchoff":
            return kirchoff_index(graph)
        case "wiener_terminal":
            return terminal_wiener_index(graph)
        case "wiener_reverse":
            return reverse_wiener_index(graph)
        case "wiener_hyper":
            return hyper_wiener_index(graph)
        case "harary":
            return harary_index(graph)
if __name__ == "__main__":
    main()