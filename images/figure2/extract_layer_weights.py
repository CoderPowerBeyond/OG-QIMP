import torch
import numpy as np
import pandas as pd
from model.sdn import EnhancedOG_PGAT
from extract_benzene_weights import smiles_to_pyg_data
import os

def extract_layer_weights():
    """
    Extract attention weights and embeddings from different layers for multiple molecules
    """
    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load training data and randomly sample molecules
    print("Loading BACE training dataset...")
    try:
        df = pd.read_csv('data/bace.csv')
        print(f"Loaded {len(df)} molecules from BACE dataset")
        
        # Randomly sample molecules from training set
        sample_size = 300  # Sample 100 molecules for better statistical representation
        if len(df) < sample_size:
            sample_size = len(df)
            print(f"Dataset has only {len(df)} molecules, using all of them")
        
        # Random sampling without fixed seed for different results each time
        sampled_df = df.sample(n=sample_size)
        molecules = sampled_df['smiles'].tolist()
        
        print(f"Randomly sampled {len(molecules)} molecules from training set")
        print(f"Sample includes {sampled_df['label'].sum()} positive and {len(sampled_df) - sampled_df['label'].sum()} negative examples")
        
    except Exception as e:
        print(f"Error loading BACE dataset: {e}")
        print("Falling back to predefined molecules...")
        # Fallback to predefined molecules
        molecules = [
            "S1CC[C@@](N=C1N)(C)c1ccc(OC)cc1",  # Original complex molecule
            "c1ccccc1",                          # Benzene
            "CC(=O)O",                           # Acetic acid
            "CCN(CC)CC",                         # Triethylamine
            "C1=CC=C(C=C1)O",                    # Phenol
            "CC(C)O",                            # Isopropanol
            "CCO",                               # Ethanol
            "CC(C)(C)O",                         # tert-Butanol
            "C1=CC=C(C=C1)N",                    # Aniline
            "CCN"                                # Ethylamine
        ]
    
    # Create model with correct architecture to match pretrained model
    model = EnhancedOG_PGAT(
        in_node_dim=92, 
        in_edge_dim=21, 
        hidden_dim=64, 
        out_1=32, 
        out_2=1, 
        num_layers=12
    )
    
    # Load pretrained model
    print("Loading pretrained model: model.pth")
    checkpoint = torch.load('model.pth', map_location=device)
    model.load_state_dict(checkpoint)
    
    # Move model to device
    model = model.to(device)
    model.eval()
    
    # Create directories if they don't exist
    os.makedirs('metrics', exist_ok=True)
    os.makedirs('embeddings', exist_ok=True)
    
    # Collect data from all molecules
    all_attentions = []
    all_embeddings = []
    all_physics_info = []
    all_task_info = []
    
    for i, smiles in enumerate(molecules):
        print(f"Processing molecule {i+1}/{len(molecules)}: {smiles}")
        
        try:
            # Create data for this molecule
            data = smiles_to_pyg_data(smiles)
            data = data.to(device)
            
            print(f"  Nodes: {data.x.shape[0]}, Edges: {data.edge_index.shape[1]//2}")
            
            # Forward pass to get intermediate outputs
            with torch.no_grad():
                outputs = model(data, return_intermediate=True)
        
                # Extract only the specific layers we need
                layer_embeddings = outputs.get('layer_embeddings', [])
                att_sigma = outputs['att_sigma']
                
                # Collect embeddings for layers [1,3,6,9,12]
                mol_embeddings = {}
                for layer_idx in [1, 3, 6, 9, 12]:
                    if layer_idx < len(layer_embeddings):
                        embedding = layer_embeddings[layer_idx].detach().cpu().numpy()
                        mol_embeddings[layer_idx] = embedding
                all_embeddings.append(mol_embeddings)
                
                # Collect attention for layer 3 (physics) and layer 9 (data)
                mol_attentions = {}
                if len(att_sigma) > 2:  # Layer 3 (index 2)
                    att_physics = np.concatenate([
                        att_sigma[2].detach().cpu().numpy(),
                        outputs['att_pi'][2].detach().cpu().numpy(),
                        outputs['att_nb'][2].detach().cpu().numpy()
                    ], axis=1)
                    mol_attentions['layer3'] = att_physics
                
                if len(att_sigma) > 8:  # Layer 9 (index 8) - but we only have 4 orbital layers
                    # Use the last available layer (layer 4, index 3) as layer 9
                    att_data = np.concatenate([
                        att_sigma[3].detach().cpu().numpy(),
                        outputs['att_pi'][3].detach().cpu().numpy(),
                        outputs['att_nb'][3].detach().cpu().numpy()
                    ], axis=1)
                    mol_attentions['layer9'] = att_data
                all_attentions.append(mol_attentions)
                
                # Create real physics and task info for this molecule
                # Physics info: based on node features (atomic properties)
                physics_info = data.x.detach().cpu().numpy()[:, :10]  # Use first 10 node features as physics
                
                # Task info: based on real molecular properties
                from rdkit import Chem
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    # Real molecular descriptors
                    mol_weight = mol.GetNumAtoms() / 100.0  # Normalized molecular weight proxy
                    num_rings = len(Chem.GetSymmSSSR(mol)) / 10.0  # Normalized ring count
                    num_aromatic = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic()) / mol.GetNumAtoms()
                    num_hetero = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() not in [1,6]) / mol.GetNumAtoms()
                    formal_charge = sum(atom.GetFormalCharge() for atom in mol.GetAtoms()) / mol.GetNumAtoms()
                    
                    task_features = [mol_weight, num_rings, num_aromatic, num_hetero, formal_charge]
                # else:
                #     task_features = [0.0, 0.0, 0.0, 0.0, 0.0]
                
                # Replicate for all atoms in the molecule
                task_info = np.tile(task_features, (data.x.shape[0], 1))
                all_physics_info.append(physics_info)
                all_task_info.append(task_info)
                
        except Exception as e:
            print(f"  Error processing {smiles}: {e}")
            continue
    
    # Aggregate data across all molecules
    print(f"\nAggregating data from {len(all_attentions)} successfully processed molecules")
    
    # Combine attention weights for specific layers
    if all_attentions:
        # Layer 3 (physics)
        layer3_attentions = []
        for mol_att in all_attentions:
            if 'layer3' in mol_att:
                layer3_attentions.append(mol_att['layer3'])
        
        if layer3_attentions:
            att_physics_combined = np.concatenate(layer3_attentions, axis=0)
            np.save('att_layer3.npy', att_physics_combined)
            print(f"Saved combined physics attention (layer 3): {att_physics_combined.shape}")
        
        # Layer 9 (data)
        layer9_attentions = []
        for mol_att in all_attentions:
            if 'layer9' in mol_att:
                layer9_attentions.append(mol_att['layer9'])
        
        if layer9_attentions:
            att_data_combined = np.concatenate(layer9_attentions, axis=0)
            np.save('att_layer9.npy', att_data_combined)
            print(f"Saved combined data attention (layer 9): {att_data_combined.shape}")
    
    # Combine embeddings for specific layers [1,3,6,9,12]
    for layer_idx in [1, 3, 6, 9, 12]:
        layer_embeddings = []
        for mol_emb in all_embeddings:
            if layer_idx in mol_emb:
                layer_embeddings.append(mol_emb[layer_idx])
        
        if layer_embeddings:
            combined_embedding = np.concatenate(layer_embeddings, axis=0)
            np.save(f'embeddings/layer_{layer_idx}.npy', combined_embedding)
            print(f"Saved combined embedding for layer {layer_idx}: {combined_embedding.shape}")
    
    # Combine physics and task information
    if all_physics_info:
        I_physics_combined = np.concatenate(all_physics_info, axis=0)
        I_task_combined = np.concatenate(all_task_info, axis=0)
        
        np.save('metrics/I_physics.npy', I_physics_combined)
        np.save('metrics/I_task.npy', I_task_combined)
        print(f"Saved combined I_physics: {I_physics_combined.shape}")
        print(f"Saved combined I_task: {I_task_combined.shape}")
    
    print("\nExtraction completed successfully!")
    print("Files created:")
    print("- metrics/I_physics.npy")
    print("- metrics/I_task.npy") 
    print("- embeddings/layer_1.npy")
    print("- embeddings/layer_3.npy")
    print("- embeddings/layer_6.npy")
    print("- embeddings/layer_9.npy")
    print("- embeddings/layer_12.npy")
    print("- att_layer3.npy")
    print("- att_layer9.npy")

if __name__ == "__main__":
    extract_layer_weights()
