import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, FancyBboxPatch
import matplotlib.patches as mpatches

# Set style
plt.style.use('default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']

# Create figure with subplots - now in horizontal layout
fig = plt.figure(figsize=(14, 7))

# Define colors for each pooling mechanism
colors = {
    'set2set': '#2E86AB',      # Deep blue
    'attention': '#A23B72',     # Purple
    'orbital': '#F18F01',       # Orange
    'reactivity': '#0fa06f'     # Red
}

# ========== Part 1: Contribution Analysis (Left panel) - Now with horizontal bars ==========
ax1 = plt.subplot(2, 5, (1, 2))

# Data for stacked bar chart showing contribution percentages
properties = ['Toxicity', 'Binding\nAffinity']
set2set_contrib = [26, 61]
orbital_contrib = [55, 22]
reactivity_contrib = [19, 17]

y = np.arange(len(properties))
height = 0.3  # 增大柱子的高度以减小间隙 (原为 0.15)
y_offset = -height/2  # 添加垂直偏移使柱子居中

# 创建横向堆叠柱状图（使用 y_offset 调整位置）
p1 = ax1.barh(y + y_offset, set2set_contrib, height, label='Set2Set', color=colors['set2set'], alpha=0.9)
p2 = ax1.barh(y + y_offset, orbital_contrib, height, left=set2set_contrib,
              label='Orbital', color=colors['orbital'], alpha=0.9)
p3 = ax1.barh(y + y_offset, reactivity_contrib, height,
              left=np.array(set2set_contrib)+np.array(orbital_contrib),
              label='Reactivity', color=colors['reactivity'], alpha=0.9)

ax1.set_xlabel('Contribution (%)', fontsize=14, fontweight='bold')
ax1.set_ylabel('Molecular Properties', fontsize=10, fontweight='bold')
ax1.set_title('Pooling Mechanism Contributions', fontsize=14, fontweight='bold', pad=15, loc='left')
ax1.text(-0.23, 1.2, 'a', transform=ax1.transAxes, fontsize=18, fontweight='bold', ha='center', va='center')
ax1.set_yticks(y)
ax1.set_yticklabels(properties, fontsize=14)
ax1.tick_params(axis='x', labelsize=14)
ax1.legend(loc='center', fontsize=12, ncol=2, bbox_to_anchor=(0.7, 0.5))
ax1.grid(True, alpha=0.3, axis='x')
ax1.set_xlim(0, 100)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# 修正标签位置（使用 y_offset）
for i in range(len(properties)):
    cumulative = 0
    for contrib, color_key in zip([set2set_contrib, orbital_contrib, reactivity_contrib],
                                   ['set2set', 'orbital', 'reactivity']):
        if contrib[i] > 5:  # Only show if contribution is significant
            ax1.text(cumulative + contrib[i]/2, y[i] + y_offset, f'{contrib[i]}%', 
                    ha='center', va='center', color='white', fontweight='bold', fontsize=12)
        cumulative += contrib[i]

# ========== Part 2: Individual Pooling Visualizations ==========

# Helper function to create molecular-like visualization
def draw_molecule_schema(ax, highlight_type='global'):
    np.random.seed(42)
    # Node positions (simulate benzene ring + side groups)
    angles = np.linspace(0, 2*np.pi, 7)[:-1]
    ring_x = np.cos(angles) * 0.3
    ring_y = np.sin(angles) * 0.3
    
    # Add some external atoms
    ext_x = np.array([0.6, -0.6, 0, 0.7])
    ext_y = np.array([0.2, -0.1, 0.7, -0.5])
    
    all_x = np.concatenate([ring_x, ext_x])
    all_y = np.concatenate([ring_y, ext_y])
    
    # Draw edges with different styles based on type
    edge_styles = []
    for i in range(6):
        if i in [1, 3, 5]:  # Double bonds
            ax.plot([ring_x[i], ring_x[(i+1)%6]], [ring_y[i], ring_y[(i+1)%6]], 
                    'gray', alpha=0.4, linewidth=3)
            ax.plot([ring_x[i], ring_x[(i+1)%6]], [ring_y[i], ring_y[(i+1)%6]], 
                    'white', alpha=0.8, linewidth=1)
        else:
            ax.plot([ring_x[i], ring_x[(i+1)%6]], [ring_y[i], ring_y[(i+1)%6]], 
                    'gray', alpha=0.3, linewidth=1.5)
    
    # Connect external atoms
    ax.plot([ring_x[0], ext_x[0]], [ring_y[0], ext_y[0]], 'gray', alpha=0.3, linewidth=1.5)
    ax.plot([ring_x[2], ext_x[1]], [ring_y[2], ext_y[1]], 'gray', alpha=0.3, linewidth=1.5)
    ax.plot([ring_x[3], ext_x[2]], [ring_y[3], ext_y[2]], 'gray', alpha=0.3, linewidth=1.5)
    ax.plot([ring_x[5], ext_x[3]], [ring_y[5], ext_y[3]], 'gray', alpha=0.3, linewidth=1.5)
    
    # Highlight nodes based on pooling type
    if highlight_type == 'global':
        # Set2Set - all nodes contribute equally
        sizes = np.ones(len(all_x)) * 250
        colors_node = [colors['set2set']] * len(all_x)
        alphas = np.ones(len(all_x)) * 0.7
        
        # Add flow arrows to show global aggregation
        for angle in np.linspace(0, 2*np.pi, 8)[:-1]:
            ax.annotate('', xy=(0, 0), xytext=(0.8*np.cos(angle), 0.8*np.sin(angle)),
                       arrowprops=dict(arrowstyle='->', color=colors['set2set'], 
                                     alpha=0.3, lw=1))
                                     
    elif highlight_type == 'attention':
        # Attention - some nodes more important
        np.random.seed(42)
        importance = np.random.uniform(0.2, 1.0, len(all_x))
        sizes = 100 + importance * 300
        colors_node = [colors['attention']] * len(all_x)
        alphas = 0.3 + importance * 0.7
        
        # Add attention weights visualization
        for i in range(3):  # Show top 3 important nodes
            idx = np.argsort(importance)[-i-1]
            circle = plt.Circle((all_x[idx], all_y[idx]), 0.15, 
                               color=colors['attention'], alpha=0.2)
            ax.add_patch(circle)
            
    elif highlight_type == 'orbital':
        # Orbital - highlight HOMO/LUMO regions
        sizes = np.ones(len(all_x)) * 200
        colors_node = []
        for i in range(len(all_x)):
            if i in [0, 2, 4]:  # HOMO nodes
                colors_node.append(colors['orbital'])
            elif i in [1, 3, 5]:  # LUMO nodes
                colors_node.append('#FFB84D')  # Lighter orange for LUMO
            else:
                colors_node.append('lightgray')
        alphas = np.ones(len(all_x))
        alphas[6:] = 0.3  # External atoms less visible
        
        # Add orbital lobes visualization
        for i in [0, 2, 4]:
            ellipse = plt.matplotlib.patches.Ellipse((ring_x[i], ring_y[i]), 
                                                    0.25, 0.15, angle=i*60,
                                                    color=colors['orbital'], alpha=0.15)
            ax.add_patch(ellipse)
            
    else:  # reactivity
        # Reactivity - highlight reactive sites
        sizes = np.ones(len(all_x)) * 180
        colors_node = []
        alphas = []
        reactive_sites = [1, 4, 7, 9]  # Simulate reactive sites
        for i in range(len(all_x)):
            if i in reactive_sites:
                colors_node.append(colors['reactivity'])
                alphas.append(1.0)
                # Add reaction site markers
                ax.plot(all_x[i], all_y[i], 'x', markersize=12, 
                       color='darkred', markeredgewidth=2)
            else:
                colors_node.append('lightgray')
                alphas.append(0.3)
    
    # Draw nodes
    for i in range(len(all_x)):
        ax.scatter(all_x[i], all_y[i], s=sizes[i], c=[colors_node[i]], 
                  alpha=alphas[i], edgecolors='black', linewidth=0.5, zorder=5)
    
    # Add atom labels for clarity
    atom_labels = ['C', 'C', 'C', 'C', 'C', 'C', 'O', 'N', 'C', 'Cl']
    for i in range(len(all_x)):
        if highlight_type == 'reactivity' and i in [1, 4, 7, 9]:
            color = 'white'
            weight = 'bold'
        else:
            color = 'black'
            weight = 'normal'
        ax.text(all_x[i], all_y[i], atom_labels[i], ha='center', va='center', 
               fontsize=8, color=color, weight=weight, zorder=10)
    
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect('equal')
    ax.axis('off')

# Set2Set Pooling
ax2 = plt.subplot(2, 5, 3)
ax2.set_position([ax2.get_position().x0 - 0.05, ax2.get_position().y0, ax2.get_position().width, ax2.get_position().height])
draw_molecule_schema(ax2, 'global')
ax2.set_title('Set2Set', fontsize=14, fontweight='bold', color=colors['set2set'])
ax2.text(0.2, 1.06, 'c', transform=ax2.transAxes, fontsize=18, fontweight='bold', ha='center', va='center')

# Orbital Pooling
ax3 = plt.subplot(2, 5, 4)
draw_molecule_schema(ax3, 'orbital')
ax3.set_title('Orbital', fontsize=14, fontweight='bold', color=colors['orbital'])
ax3.text(0.2, 1.06, 'd', transform=ax3.transAxes, fontsize=18, fontweight='bold', ha='center', va='center')
homo_patch = mpatches.Patch(color=colors['orbital'], label='HOMO', alpha=0.8)
lumo_patch = mpatches.Patch(color='#FFB84D', label='LUMO', alpha=0.8)
ax3.legend(handles=[homo_patch, lumo_patch], loc='upper right', fontsize=10, framealpha=0.9)

# Reactivity Pooling
ax4 = plt.subplot(2, 5, 9)
draw_molecule_schema(ax4, 'reactivity')
ax4.set_title('Reactivity', fontsize=14, fontweight='bold', color=colors['reactivity'])
ax4.text(0.2, 1.06, 'e', transform=ax4.transAxes, fontsize=18, fontweight='bold', ha='center', va='center')
ax4.set_position([ax4.get_position().x0, ax4.get_position().y0 + 0.1, ax4.get_position().width, ax4.get_position().height])

# ========== Performance comparison plot - below the first figure ==========
ax5 = plt.subplot(2, 5, (6, 8))

# Performance data
pooling_methods = ['Combined\n(Ours)', 'Set2Set\nOnly', 
                   'Orbital\nOnly', 'Reactivity\nOnly', 'Standard\nMean', 'Standard\nMax']
performance = [0.914, 0.876, 0.876, 0.874, 0.879, 0.894]
std_dev = [0.016, 0.028, 0.038, 0.005, 0.002, 0.012]

y_pos = np.arange(len(pooling_methods))
bars = ax5.barh(y_pos, performance, xerr=std_dev, capsize=5, 
                color=['#FF6B6B', colors['set2set'], 
                       colors['orbital'], colors['reactivity'], 'gray', 'gray'],
                alpha=0.8, edgecolor='black', linewidth=1)

# Highlight the best performance
bars[0].set_alpha(1.0)
bars[0].set_linewidth(2)

ax5.set_ylabel('Pooling Method', fontsize=10, fontweight='bold')
ax5.set_xlabel('Performance (ROC-AUC)', fontsize=14, fontweight='bold')
ax5.set_title('Performance Comparison of Pooling Mechanisms', fontsize=14, fontweight='bold', loc='left')
ax5.text(-0.125, 1.08, 'b', transform=ax5.transAxes, fontsize=18, fontweight='bold', ha='center', va='center')
ax5.set_yticks(y_pos)
ax5.set_yticklabels(pooling_methods, fontsize=10)
ax5.tick_params(axis='x', labelsize=14)
ax5.set_xlim(0.75, 0.95)
ax5.grid(True, alpha=0.3, axis='x')
ax5.spines['top'].set_visible(False)
ax5.spines['right'].set_visible(False)

# Add value labels on bars
for i, (bar, val) in enumerate(zip(bars, performance)):
    ax5.text(val + 0.005, bar.get_y() + bar.get_height()/2 - 0.2, f'{val:.3f}', 
            ha='left', va='center', fontweight='bold', fontsize=12)

# Add significance marker for the best performance
ax5.text(0.935, -0.2, '*', ha='left', va='center', fontsize=12, fontweight='bold', color='darkgreen')

# Invert y-axis to have best performance on top
ax5.invert_yaxis()

# Add main title
# fig.suptitle('Multi-Scale Pooling Mechanism Contributions and Performance', fontsize=16, fontweight='bold', x=0.37, y=0.98, ha='center')

plt.subplots_adjust(left=0.05, right=0.95, top=0.96, bottom=0.02, hspace=-0.3, wspace=0.1)
# Make second chart shorter
ax5.set_position([ax5.get_position().x0, 0.2, ax5.get_position().width, 0.3])
# Manually adjust subplot positions to make first chart flatter
ax1.set_position([0.05, 0.65, 0.3, 0.2])

# Save figure
plt.savefig('multiscale_pooling_contribution_horizontal.pdf', dpi=300, bbox_inches='tight')
plt.savefig('multiscale_pooling_contribution_horizontal.png', dpi=300, bbox_inches='tight')
plt.show()