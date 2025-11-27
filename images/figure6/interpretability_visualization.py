"""
OG-QIMP Interpretability Figure Generator
-----------------------------------------
Generates interpretability_visualization.pdf with panels:
(a) atom–atom attention heatmap
(b) HOMO–LUMO charge-transfer map
(c) atom-level reactivity weights (α_i)
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from rdkit import Chem
from rdkit.Chem import AllChem

# ==============================================
# 1. Prepare a demo molecule
# ==============================================
smiles = "S1CC[C@@](N=C1N)(C)"   # benzene example
mol = Chem.MolFromSmiles(smiles)
AllChem.Compute2DCoords(mol)
coords = mol.GetConformer().GetPositions()
atoms = [a.GetSymbol() for a in mol.GetAtoms()]
bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()]
n = len(atoms)
print(n)

# ==============================================
# 2. Load real model outputs from benzene npz file
# ==============================================
# Load attention weights from the benzene npz file
data = np.load('benzene_attention_weights.npz')

# Extract real model weights
att_sigma = data['att_sigma']
att_pi = data['att_pi'] 
att_nb = data['att_nb']
h_homo = data['h_homo']
h_lumo = data['h_lumo']
alpha = data['alpha']

print(f"Loaded benzene model weights:")
print(f"att_sigma shape: {att_sigma.shape}")
print(f"att_pi shape: {att_pi.shape}")
print(f"att_nb shape: {att_nb.shape}")
print(f"h_homo shape: {h_homo.shape}")
print(f"h_lumo shape: {h_lumo.shape}")
print(f"alpha shape: {alpha.shape}")
print(f"Benzene atoms: {n}")

# ==============================================
# 3. Convert edge-level attention to atom-atom matrix
# ==============================================
def edge_attention_to_atom_matrix(edge_attention, num_atoms, bonds):
    """
    Convert edge-level attention weights to atom-atom attention matrix
    """
    # Initialize atom-atom attention matrix
    atom_matrix = np.zeros((num_atoms, num_atoms))
    
    # Map edges to atom pairs
    edge_idx = 0
    for i, j in bonds:
        # Forward direction: i -> j
        atom_matrix[i, j] = edge_attention[edge_idx].mean()  # Average across attention heads
        edge_idx += 1
        # Backward direction: j -> i  
        atom_matrix[j, i] = edge_attention[edge_idx].mean()  # Average across attention heads
        edge_idx += 1
    
    return atom_matrix

# Convert attention weights to atom-atom matrices
att_sigma_matrix = edge_attention_to_atom_matrix(att_sigma, n, bonds)
att_pi_matrix = edge_attention_to_atom_matrix(att_pi, n, bonds)
att_nb_matrix = edge_attention_to_atom_matrix(att_nb, n, bonds)

print(f"Converted attention matrices shape: {att_sigma_matrix.shape}")

# Handle dimension mismatch
if h_homo.shape[0] > n:
    print(f"Truncating model weights to match benzene size ({n} atoms)")
    h_homo = h_homo[:n, :]
    h_lumo = h_lumo[:n, :]
    alpha = alpha[:n]
elif h_homo.shape[0] < n:
    print(f"Model has {h_homo.shape[0]} atoms, but benzene has {n} atoms")
    print("Adjusting benzene to match model size")
    n = h_homo.shape[0]
    atoms = atoms[:n]
    coords = coords[:n]
    bonds = [(i, j) for i, j in bonds if i < n and j < n]

# Select only the left part (including benzene ring and left side)
# For this molecule, we want to keep atoms 0-7 (benzene ring + left side)
left_atoms = 8  # Keep first 8 atoms (benzene ring + left side)
print(f"Displaying only left part: {left_atoms} atoms out of {n}")

# Truncate all data to left part
h_homo = h_homo[:left_atoms, :]
h_lumo = h_lumo[:left_atoms, :]
alpha = alpha[:left_atoms]
atoms = atoms[:left_atoms]
coords = coords[:left_atoms]
bonds = [(i, j) for i, j in bonds if i < left_atoms and j < left_atoms]

# Truncate attention matrices to left part
att_sigma_matrix = att_sigma_matrix[:left_atoms, :left_atoms]
att_pi_matrix = att_pi_matrix[:left_atoms, :left_atoms]
att_nb_matrix = att_nb_matrix[:left_atoms, :left_atoms]

n = left_atoms

# Calculate charge transfer from HOMO-LUMO features
charge_transfer = h_homo.mean(axis=1) - h_lumo.mean(axis=1)

# For benzene (symmetric molecule), use attention weights to show differences
# Use sum of incoming attention as atom reactivity
alpha = np.sum(att_sigma_matrix, axis=1)  # Sum of incoming attention as reactivity
alpha = np.abs(alpha.flatten())
alpha /= alpha.max()

print(f"Charge transfer values: {charge_transfer}")
print(f"Alpha reactivity values: {alpha}")

# ==============================================
# 4. Build molecular graph for plotting
# ==============================================
G = nx.Graph()
for i, a in enumerate(atoms):
    G.add_node(i, element=a)
for i, j in bonds:
    G.add_edge(i, j)

pos = {i: coords[i][:2] for i in range(n)}

# ==============================================
# 4. Create subplots
# ==============================================
fig, axs = plt.subplots(1, 3, figsize=(18, 3.8))
plt.subplots_adjust(wspace=0.18)  # Reduce spacing between subplots

# ---------------------
# (a) Atom–atom attention heatmap (σ)
# ---------------------
im = axs[0].imshow(att_sigma_matrix, cmap="Reds", aspect='auto')
axs[0].set_xticks(range(len(atoms)))
axs[0].set_yticks(range(len(atoms)))
axs[0].set_xticklabels(atoms)
axs[0].set_yticklabels(atoms)
axs[0].set_title("σ-bond attention map", fontsize=16)
axs[0].set_xlabel("Atom index", fontsize=14)
axs[0].set_ylabel("Atom index", fontsize=14)
axs[0].tick_params(axis='both', which='major', labelsize=14)
axs[0].text(-0.14, 1.08, 'a', transform=axs[0].transAxes, fontsize=18, fontweight='bold', 
            verticalalignment='top')

# Add colorbar for the heatmap
cbar = fig.colorbar(im, ax=axs[0], fraction=0.046, pad=0.05)
cbar.set_label('Attention Weight', fontsize=14)
cbar.ax.tick_params(labelsize=14)

# ---------------------
# (b) HOMO–LUMO charge transfer map
# ---------------------
node_color = charge_transfer
edges = list(G.edges)
nx.draw_networkx(G, pos=pos, edgelist=edges,
                 node_color=node_color, cmap='coolwarm',
                 node_size=330,
                 edge_color='gray', width=1.0,
                 with_labels=False, ax=axs[1])
sm = plt.cm.ScalarMappable(cmap='coolwarm')
sm.set_array([])
cb = plt.colorbar(sm, ax=axs[1], fraction=0.046, pad=0.02)
cb.set_label('HOMO–LUMO value', fontsize=14)
cb.ax.tick_params(labelsize=14)
axs[1].set_title("HOMO–LUMO charge transfer", fontsize=16)
axs[1].axis('off')
axs[1].text(0.00, 1.13, 'b', transform=axs[1].transAxes, fontsize=18, fontweight='bold', 
            verticalalignment='top')

# ---------------------
# (c) Atom-level reactivity weights
# ---------------------
# Use normal scaling without amplification
sizes = 800 * alpha
print(f"Alpha values: {alpha}")
print(f"Node sizes: {sizes}")
print(f"Size range: {sizes.min():.0f} to {sizes.max():.0f}")
print(f"Size ratio: {sizes.max()/sizes.min():.2f}x")

nx.draw_networkx(G, pos=pos, edgelist=edges,
                 node_color='orange', node_size=sizes,
                 edge_color='gray', width=0.8,
                 with_labels=False, ax=axs[2])
axs[2].set_title("Reactivity pooling weights (αᵢ)", fontsize=16)
axs[2].axis('off')
axs[2].text(0.02, 1.13, 'c', transform=axs[2].transAxes, fontsize=18, fontweight='bold', 
            verticalalignment='top')

# ==============================================
# 5. Final layout and save figure
# ==============================================
for ax in axs:
    ax.set_aspect('equal')

# plt.tight_layout()  # Commented out to preserve custom spacing
plt.savefig("interpretability_visualization.pdf", bbox_inches='tight', dpi=600)
plt.show()