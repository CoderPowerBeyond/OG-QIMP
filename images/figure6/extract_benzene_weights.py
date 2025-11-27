#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract attention weights from SDN model for benzene molecule
"""

import torch
import numpy as np
from model.sdn import EnhancedOG_PGAT
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import AllChem

def smiles_to_pyg_data(smiles, node_dim=92, edge_dim=21):
    """
    Convert SMILES to PyTorch Geometric Data object with deterministic features
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    
    # Get number of atoms
    num_atoms = mol.GetNumAtoms()
    
    # Create deterministic node features based on atom properties
    node_features = []
    for atom in mol.GetAtoms():
        # Create features based on atom properties (deterministic)
        atom_num = atom.GetAtomicNum()
        formal_charge = atom.GetFormalCharge()
        hybridization = atom.GetHybridization()
        is_aromatic = atom.GetIsAromatic()
        
        # Create a simple, reliable feature vector
        # Use basic one-hot encoding and simple normalization
        feat = [
            float(atom_num),  # Raw atomic number (most important feature)
            float(formal_charge),  # Raw formal charge
            float(int(hybridization)),  # Raw hybridization
            float(is_aromatic),  # Aromaticity: 0 or 1
        ]
        
        # Pad to node_dim with deterministic values
        while len(feat) < node_dim:
            feat.append(np.sin(len(feat) * atom_num) / 10.0)  # Deterministic padding
        
        node_features.append(feat[:node_dim])
    
    node_features = torch.tensor(node_features, dtype=torch.float32)
    
    # Create edge index and edge features
    edge_list = []
    edge_features = []
    
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_list.append([i, j])
        edge_list.append([j, i])  # Add reverse edge
        
        # Create deterministic edge features based on bond properties
        bond_type = bond.GetBondType()
        is_conjugated = bond.GetIsConjugated()
        
        edge_feat = [
            float(int(bond_type)),  # Raw bond type: 1, 2, 3
            float(is_conjugated),  # Conjugation: 0 or 1
        ]
        
        # Pad to edge_dim with deterministic values
        while len(edge_feat) < edge_dim:
            edge_feat.append(np.cos(len(edge_feat) * (i + j)) / 10.0)  # Deterministic padding
        
        edge_feat = edge_feat[:edge_dim]
        edge_features.append(edge_feat)
        edge_features.append(edge_feat)  # Same features for both directions
    
    edge_index = torch.tensor(edge_list).t().contiguous()
    edge_attr = torch.tensor(edge_features, dtype=torch.float32) if edge_features else None
    
    # Create batch (single molecule)
    batch = torch.zeros(num_atoms, dtype=torch.long)
    
    return Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, batch=batch)

def extract_weights_for_benzene():
    """
    Extract attention weights for benzene molecule
    """
    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # SMILES for benzene
    smiles = "S1CC[C@@](N=C1N)(C)c1ccc(OC)cc1"
    print(f"Processing molecule: {smiles}")
    
    # Create model with correct architecture to match pretrained model
    model = EnhancedOG_PGAT(
        in_node_dim=92, 
        in_edge_dim=21, 
        hidden_dim=64, 
        out_1=32, 
        out_2=1, 
        num_layers=2  # Pre-trained model has 2 orbital_guided_layers
    )
    
    # Load pretrained model
    print("Loading pretrained model: model.pth")
    checkpoint = torch.load('model.pth', map_location=device)
    model.load_state_dict(checkpoint)
    
    # Move model to device
    model = model.to(device)
    
    # Set to evaluation mode
    model.eval()
    print("Model loaded successfully!")
    
    # Convert SMILES to PyTorch Geometric data
    data = smiles_to_pyg_data(smiles)
    print(f"Data created - Nodes: {data.x.shape[0]}, Edges: {data.edge_index.shape[1]}")
    print(f"Node features shape: {data.x.shape}")
    print(f"Edge features shape: {data.edge_attr.shape if data.edge_attr is not None else 'None'}")
    
    # Move data to device
    data = data.to(device)
    
    # Forward pass with intermediate outputs
    with torch.no_grad():
        outputs = model(data, return_intermediate=True)
    
    # Extract attention weights
    if isinstance(outputs, dict) and 'att_sigma' in outputs:
        att_sigma = outputs["att_sigma"][-1].detach().cpu().numpy()
        att_pi = outputs["att_pi"][-1].detach().cpu().numpy()
        att_nb = outputs["att_nb"][-1].detach().cpu().numpy()
        h_homo = outputs["homo_feats"].detach().cpu().numpy()
        h_lumo = outputs["lumo_feats"].detach().cpu().numpy()
        alpha = outputs["pool_weights"].detach().cpu().numpy()
        
        print(f"Attention weights extracted successfully!")
        print(f"att_sigma shape: {att_sigma.shape}")
        print(f"att_pi shape: {att_pi.shape}")
        print(f"att_nb shape: {att_nb.shape}")
        print(f"h_homo shape: {h_homo.shape}")
        print(f"h_lumo shape: {h_lumo.shape}")
        print(f"alpha shape: {alpha.shape}")
        
        # Save to npz file
        np.savez('benzene_attention_weights.npz',
                att_sigma=att_sigma,
                att_pi=att_pi,
                att_nb=att_nb,
                h_homo=h_homo,
                h_lumo=h_lumo,
                alpha=alpha)
        
        print("Weights saved to benzene_attention_weights.npz")
        
        return {
            'att_sigma': att_sigma,
            'att_pi': att_pi,
            'att_nb': att_nb,
            'h_homo': h_homo,
            'h_lumo': h_lumo,
            'alpha': alpha
        }
    else:
        print("Failed to extract attention weights")
        return None

if __name__ == "__main__":
    weights = extract_weights_for_benzene()
    if weights:
        print("\nExtraction completed successfully!")
        print("You can now use benzene_attention_weights.npz in fig_interpre.py")
    else:
        print("Extraction failed!")
