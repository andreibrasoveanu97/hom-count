import streamlit as st
from Models.encoder import NodeEncoder, EdgeEncoder
from Models.gnn import GNN
from Misc.cell_encoding import CellularRingEncoding
import torch
from torch_geometric.data import Data
from torch_geometric import utils
from torch_geometric.datasets import ZINC
from rdkit import Chem
from rdkit.Chem import Draw
import numpy as np

args = {'config_file':None, 'tracking':1, 'seed':0, 'split':0, 'dataset':'ZINC', 'lr':0.001, 'lr2':0.0001,
        'batch_size':128, 'epochs':1000, 'max_params':1000000000.0, 'device':0, 'model':'GIN', 'JK':'concat',
        'graph_trafo':'CWN', 'max_struct_size':8, 'lr_scheduler':'ReduceLROnPlateau', 'lr_scheduler_decay_rate':0.5,
        'lr_scheduler_decay_steps':50, 'min_lr':1e-05, 'lr_schedule_patience':20, 'max_time':12, 'drop_out':0.5,
        'emb_dim':256, 'num_layers':5, 'num_mlp_layers':2, 'virtual_node':0, 'dummy':False, 'pooling':'mean',
        'dim_pooling':0, 'node_encoder':1, 'graph_feat':'../hom-count/Counts/GlobalFeatures/ZINC_WIENER_global.json',
        'node_broadcast':False, 'freeze_gnn':0, 'drop_feat':0, 'misalign':0, 'cliques':0, 'rings':0,
        'aggr_edge_atr':0, 'aggr_vertex_feat':0, 'explicit_pattern_enc':0, 'edge_attr_in_vertices':0,
        'use_tracking':True, 'use_virtual_node':False, 'use_node_encoder':True, 'do_freeze_gnn':False,
        'do_drop_feat':False, 'use_misaligned':False, 'use_rings':False, 'use_cliques':False,
        'use_aggr_edge_atr':False, 'use_aggr_vertex_feat':False, 'use_explicit_pattern_enc':False,
        'use_edge_attr_in_vertices':False}

# Dictionary for the zinc dataset node feature labels
node_feature_dict = {
    'C': 0, 'O': 1, 'N': 2, 'F': 3, 'C H1': 4, 'S': 5, 'Cl': 6, 'O -': 7,
    'N H1 +': 8, 'Br': 9, 'N H3 +': 10, 'N H2 +': 11, 'N +': 12, 'N -': 13,
    'S -': 14, 'I': 15, 'P': 16, 'O H1 +': 17, 'N H1 -': 18, 'O +': 19,
    'S +': 20, 'P H1': 21, 'P H2': 22, 'C H2 -': 23, 'P +': 24, 'S H1 +': 25,
    'C H1 -': 26, 'P H1 +': 27
}

# Dictionary for bond types
bond_type_dict = {
    Chem.rdchem.BondType.SINGLE: 1,
    Chem.rdchem.BondType.DOUBLE: 2,
    Chem.rdchem.BondType.TRIPLE: 3,
}

def translate_smiles(smiles):
    # Parse the SMILES string using RDKit
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        raise ValueError("Invalid SMILES string")

    node_features = []

    for atom in mol.GetAtoms():
        # Basic element symbol
        symbol = atom.GetSymbol()

        # Explicit hydrogens directly from SMILES string
        explicit_hs = atom.GetNumExplicitHs()
        # Adjust explicit hydrogens for nitrogen atoms
        formal_charge = atom.GetFormalCharge()

        feature_key = symbol

        # Add hydrogen and charge information to the key only if it is explicitly defined in the dictionary
        if explicit_hs > 0:
            possible_key = f'{symbol} H{explicit_hs}'
            feature_key = possible_key
        if formal_charge > 0:
            possible_key = f'{feature_key} +'
            if possible_key in node_feature_dict:
                feature_key = possible_key
        elif formal_charge < 0:
            possible_key = f'{feature_key} -'
            if possible_key in node_feature_dict:
                feature_key = possible_key
        # Get the corresponding label from the dictionary
        if feature_key in node_feature_dict:
            node_features.append(node_feature_dict[feature_key])
        else:
            # If the exact feature is not in the dictionary, fall back to the symbol
            if symbol in node_feature_dict:
                node_features.append(node_feature_dict[symbol])
            else:
                raise ValueError(f"Unknown feature: {feature_key}")

    return node_features


def build_data_from_smiles(smiles):
    data = utils.from_smiles(smiles, kekulize=True)

    node_features = translate_smiles(smiles)

    # Convert to torch tensors
    x = torch.tensor(node_features, dtype=torch.long).view(-1, 1)
    edge_index = data.edge_index
    edge_attr = data.edge_attr[:, 0]

    # Create Data object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    return data

def get_model(args, num_classes, num_vertex_features, num_tasks):
    node_feature_dims = []

    if args['graph_trafo'] == "CWN":
        node_feature_dims = [2, 2, 2]

    model = args['model'].lower()

    if args['dataset'].lower() == "zinc" and not args['do_drop_feat']:
        node_feature_dims.append(21)
        node_feature_dims.append(1001)
        node_encoder = NodeEncoder(emb_dim=args['emb_dim'], feature_dims=node_feature_dims)
        edge_encoder = EdgeEncoder(emb_dim=args['emb_dim'], feature_dims=[4])
    else:
        node_encoder, edge_encoder = lambda x: x, lambda x: x

    max_struct_size = args['max_struct_size']
    if max_struct_size > 0 and not args['cliques']:
        max_struct_size = 2
    if model in ["gin", "gcn", "gat"]:
        return GNN(num_classes, num_tasks, args['num_layers'], args['emb_dim'],
                   gnn_type=model, virtual_node=args['use_virtual_node'], drop_ratio=args['drop_out'],
                   JK=args['JK'], graph_pooling=args['pooling'], edge_encoder=edge_encoder, node_encoder=node_encoder,
                   use_node_encoder=args['use_node_encoder'], num_mlp_layers=args['num_mlp_layers'],
                   dim_pooling=args['graph_trafo'] == "CWN")
    else:  # Probably don't need other models
        pass

    return model

def predict_logp(data_pt):
    model = get_model(args, 1, 4, 1)
    model.load_state_dict(torch.load('./Results/Models/ZINC_2024-05-27_19:35:48', map_location=torch.device('cpu')))

    model.eval()

    with torch.no_grad():
        try:
            pred = model(data_pt)
        except Exception as e:
            pred = None
            st.error(f"Error predicting the logP value for this molecule: {e}")
    return np.round(float(pred), 2)


def display_molecule(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return Draw.MolToImage(mol)


st.title("Predict Constrained Solubility of a Molecule (logP)")

transform = CellularRingEncoding(args['max_struct_size'], aggr_edge_atr=True, aggr_vertex_feat=True,
                                               explicit_pattern_enc=True, edge_attr_in_vertices=False)

smiles_input = st.text_input("Enter SMILES string:")
if st.button("Predict"):
    if smiles_input:
        try:
            mol = Chem.MolFromSmiles(smiles_input)
            if mol:

                data_pt = build_data_from_smiles(smiles_input)
                # apply the cellular ring encoding format expected by the model
                transform(data_pt)
                # Predict logP
                logp = predict_logp(data_pt)

                # Display molecule structure
                st.image(Draw.MolToImage(mol), caption="Molecule Structure")

                # Display logP prediction
                st.write(f"Predicted logP: {logp}")
            else:
                st.error("Invalid SMILES string")
        except Exception as e:
            st.error(f"Error processing SMILES string: {e}")
    else:
        st.error("Please enter a SMILES string")