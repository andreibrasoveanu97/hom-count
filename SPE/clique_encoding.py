
import torch
import graph_tool as gt
import graph_tool.topology as top
import networkx as nx
from torch_geometric.transforms import BaseTransform
from torch_geometric.data import Data


class CellularCliqueEncoding(BaseTransform):
    r"""
    Args:

            
    """
    def __init__(self, max_clique_size: int, aggr_edge_atr: bool = False, 
        aggr_vertex_feat: bool = False, explicit_pattern_enc: bool = False, edge_attr_in_vertices: bool = False):
        self.max_clique_size = max_clique_size
        self.aggr_edge_atr = aggr_edge_atr
        self.aggr_vertex_feat = aggr_vertex_feat
        self.explicit_pattern_enc = explicit_pattern_enc
        self.edge_attr_in_vertices = edge_attr_in_vertices

    def encode_with_id_maps(self, data: Data):  
        pass
        # return data, id_maps

    def __call__(self, data: Data):
        data, _ = self.encode_with_id_maps(data)
        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'