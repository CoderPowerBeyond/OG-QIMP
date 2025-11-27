import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
from scipy.interpolate import make_interp_spline

# Set Nature-style parameters with LARGER FONTS
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica Neue']
plt.rcParams['font.size'] = 16
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['xtick.major.width'] = 0.8
plt.rcParams['ytick.major.width'] = 0.8
plt.rcParams['xtick.major.size'] = 4
plt.rcParams['ytick.major.size'] = 4

# Nature color palette
colors = {
    'og_qimp': '#D62728',  # Distinctive red
    'ka_gnn': '#2CA02C',   # Green
    'standard_gnn': '#1F77B4',  # Blue
    'd_mpnn': '#FF7F0E',   # Orange
    'attentivefp': '#9467BD',  # Purple
    'causal': '#8C564B'    # Brown
}

# Create main figure with MORE SPACE between panels
fig = plt.figure(figsize=(18, 5.5))  # Increased width from 16 to 18
gs = GridSpec(1, 3, figure=fig, wspace=0.4, hspace=0.3)  # Increased wspace from 0.3 to 0.4

# ============ Panel a: Performance Curves with Confidence Bands ============
ax1 = fig.add_subplot(gs[0])

# Generate smooth data
x = np.array([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
x_smooth = np.linspace(0.3, 0.9, 300)

# Model performance data
models_data = {
    'OG-QIMP (Ours)': {'mean': np.array([68, 71, 75, 82, 88, 93, 96]), 
                       'std': np.array([2, 2.5, 2, 1.8, 1.5, 1.2, 1])},
    'KA-GNN': {'mean': np.array([63, 65, 68, 73, 78, 84, 89]),
               'std': np.array([3, 3.2, 3, 2.8, 2.5, 2.2, 2])},
    'Standard GNN': {'mean': np.array([62, 63, 65, 69, 73, 78, 83]),
                     'std': np.array([3.5, 3.5, 3.2, 3, 2.8, 2.5, 2.3])},
    'D-MPNN': {'mean': np.array([64, 66, 69, 74, 79, 85, 90]),
               'std': np.array([2.8, 2.8, 2.5, 2.3, 2, 1.8, 1.5])},
    'AttentiveFP': {'mean': np.array([65, 68, 71, 76, 81, 87, 92]),
                    'std': np.array([2.5, 2.5, 2.3, 2, 1.8, 1.5, 1.3])}
}

# Plot with smooth curves and confidence bands
for model_name, data in models_data.items():
    # Create smooth interpolation
    spl = make_interp_spline(x, data['mean'], k=3)
    mean_smooth = spl(x_smooth)
    
    spl_upper = make_interp_spline(x, data['mean'] + data['std'], k=3)
    upper_smooth = spl_upper(x_smooth)
    
    spl_lower = make_interp_spline(x, data['mean'] - data['std'], k=3)
    lower_smooth = spl_lower(x_smooth)
    
    # Determine color and style
    if 'OG-QIMP' in model_name:
        color = colors['og_qimp']
        linewidth = 2.5
        alpha_fill = 0.2
        zorder = 10
    else:
        color = colors.get(model_name.lower().replace('-', '_').replace(' ', '_'), '#888888')
        linewidth = 1.5
        alpha_fill = 0.1
        zorder = 5
    
    # Plot line and confidence band
    ax1.plot(x_smooth, mean_smooth, label=model_name, color=color, 
             linewidth=linewidth, zorder=zorder)
    ax1.fill_between(x_smooth, lower_smooth, upper_smooth, 
                     color=color, alpha=alpha_fill)

# Add gradient background for distribution shift regions
gradient = np.linspace(0, 1, 256).reshape(1, -1)
gradient = np.vstack((gradient, gradient))

extent = [0.3, 0.9, 60, 100]
ax1.imshow(gradient, extent=extent, aspect='auto', cmap='RdYlGn_r', alpha=0.15, zorder=0)

# Annotations for key retention values
ax1.annotate('91.2% retention', xy=(0.85, 95), xytext=(0.65, 97),
            arrowprops=dict(arrowstyle='->', color=colors['og_qimp'], lw=1.5, alpha=0.7),
            fontsize=15, color=colors['og_qimp'], fontweight='bold')

ax1.annotate('68.4% retention', xy=(0.35, 70), xytext=(0.3, 82),
            arrowprops=dict(arrowstyle='->', color=colors['og_qimp'], lw=1.5, alpha=0.7),
            fontsize=15, color=colors['og_qimp'], fontweight='bold')

# Styling
ax1.set_xlabel('Chemical Similarity (Tanimoto)', fontsize=17, fontweight='medium')
ax1.set_ylabel('Performance (% of Original)', fontsize=17, fontweight='medium')
ax1.set_xlim(0.28, 0.92)
ax1.set_ylim(60, 100)
ax1.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Legend with custom styling
legend = ax1.legend(loc='upper left', frameon=False, fancybox=True, 
                   shadow=False, borderpad=0.8, columnspacing=1, fontsize=15)
legend.get_frame().set_facecolor('white')
legend.get_frame().set_alpha(0.95)
legend.get_frame().set_edgecolor('#CCCCCC')

# Add distribution shift labels with better spacing
# ax1.text(0.35, 58, 'Very Different', fontsize=14, ha='center', color='#666666')
# ax1.text(0.6, 58, 'Moderate', fontsize=14, ha='center', color='#666666')
# ax1.text(0.85, 58, 'Very Similar', fontsize=14, ha='center', color='#666666')

# Set tick label sizes
ax1.tick_params(axis='both', which='major', labelsize=15)

# ============ Panel b: Performance Drop Heatmap ============
ax2 = fig.add_subplot(gs[1])

# Create performance matrix - SHORTER LABELS to prevent overlap
models = ['OG-QIMP\n(Ours)', 'AttentiveFP', 'D-MPNN', 'KA-GNN', 'Standard\nGNN']
shift_levels = ['Very Diff.\n(0.3)', 'Different\n(0.5)', 'Moderate\n(0.7)', 'Similar\n(0.9)']

# Performance drop data (100 - retention %)
perf_drops = np.array([
    [8.8, 6.2, 3.5, 1.2],   # OG-QIMP
    [12.5, 9.3, 5.8, 2.1],  # AttentiveFP
    [14.2, 10.5, 6.2, 2.5], # D-MPNN
    [18.3, 13.7, 8.1, 3.2], # KA-GNN
    [21.5, 16.8, 10.3, 4.1], # Standard GNN
    # [25.3, 19.2, 12.1, 5.3]  # CausalMol
])

# Create custom colormap
from matplotlib.colors import LinearSegmentedColormap
colors_heatmap = ['#2E7D32', '#66BB6A', '#FFF59D', '#FFB74D', '#E65100']
n_bins = 100
cmap = LinearSegmentedColormap.from_list('custom', colors_heatmap, N=n_bins)

# Plot heatmap
im = ax2.imshow(perf_drops, cmap=cmap, aspect='auto', vmin=0, vmax=30)

# Add text annotations
for i in range(len(models)):
    for j in range(len(shift_levels)):
        value = perf_drops[i, j]
        color = 'black' if value > 15 else 'black'
        text = ax2.text(j, i, f'{value:.1f}%', ha='center', va='center',
                       color=color, fontsize=16, fontweight='medium')

# Styling with ROTATED X-LABELS
ax2.set_xticks(np.arange(len(shift_levels)))
ax2.set_yticks(np.arange(len(models)))
ax2.set_xticklabels(shift_levels, fontsize=16, rotation=45, ha='right')  # ROTATED 45 degrees
ax2.set_yticklabels(models, fontsize=16)
# ax2.set_xlabel('Distribution Shift Level', fontsize=17, fontweight='medium', labelpad=10)
ax2.set_title('Performance Drop (%)', fontsize=17, fontweight='medium', pad=12)

# Add colorbar
cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.06)  # Increased pad
cbar.set_label('Drop (%)', rotation=270, labelpad=18, fontsize=16)
cbar.ax.tick_params(labelsize=14)

# Remove spines
for spine in ax2.spines.values():
    spine.set_visible(False)

# Only white lines between cells
ax2.set_xticks(np.arange(len(shift_levels)+1)-.5, minor=True)
ax2.set_yticks(np.arange(len(models)+1)-.5, minor=True)
ax2.grid(which="minor", color="white", linestyle='-', linewidth=2)
ax2.tick_params(which="minor", size=0)

# ============ Panel c: Robustness Metrics Bar Chart ============
ax3 = fig.add_subplot(gs[2])

# Data for bar chart - SHORTER LABELS
models_short = ['OG-QIMP\n(Ours)', 'KA-GNN', 'Standard\nGNN', 'D-MPNN', 'AttentiveFP']
acceptable_10 = [5.5, 31.6, 38.7, 36.2, 40.3]
critical_30 = [1.2, 18.3, 25.8, 22.1, 28.5]

x_pos = np.arange(len(models_short))
width = 0.32  # Slightly reduced width for better spacing

# Create bars with gradient effect
bars1 = ax3.bar(x_pos - width/2, acceptable_10, width, label='10% Threshold',
                color='#42A5F5', edgecolor='#1976D2', linewidth=1.5)
bars2 = ax3.bar(x_pos + width/2, critical_30, width, label='30% Threshold',
                color='#66BB6A', edgecolor='#388E3C', linewidth=1.5)

# Highlight OG-QIMP
bars1[0].set_color('#D62728')
bars1[0].set_edgecolor('#8B0000')
bars2[0].set_color('#FF6B6B')
bars2[0].set_edgecolor('#8B0000')

# Add value labels on bars with better positioning
for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
    height1 = bar1.get_height()
    height2 = bar2.get_height()
    
    # Style for OG-QIMP
    weight = 'bold' if i == 0 else 'normal'
    color = '#8B0000' if i == 0 else '#333333'
    
    ax3.text(bar1.get_x() + bar1.get_width()/2., height1 + 0.8,
            f'{height1:.1f}%', ha='center', va='bottom', fontsize=14,
            fontweight=weight, color=color)
    ax3.text(bar2.get_x() + bar2.get_width()/2., height2 + 0.8,
            f'{height2:.1f}%', ha='center', va='bottom', fontsize=14,
            fontweight=weight, color=color)

# Add threshold lines
ax3.axhline(y=10, color='#FF9800', linestyle='--', linewidth=1.5, alpha=0.7, label='Acceptable (10%)')
ax3.axhline(y=30, color='#F44336', linestyle='--', linewidth=1.5, alpha=0.7, label='Critical (30%)')

# Add annotation with better positioning
ax3.annotate('Best robustness', xy=(0, 5.5), xytext=(0.1, 18),
            arrowprops=dict(arrowstyle='->', color='#D62728', lw=1.5),
            fontsize=14, color='#D62728', fontweight='bold')

# Styling with ROTATED X-LABELS
# ax3.set_xlabel('Model', fontsize=17, fontweight='medium', labelpad=10)
ax3.set_ylabel('Performance Drop (%)', fontsize=17, fontweight='medium')
ax3.set_title('Robustness Under Distribution Shift', fontsize=17, fontweight='medium', pad=12)
ax3.set_xticks(x_pos)
ax3.set_xticklabels(models_short, fontsize=14, rotation=45, ha='right')  # ROTATED 45 degrees
ax3.set_ylim(0, 50)  # INCREASED from 45 to 50 to make more room for legend
ax3.grid(True, axis='y', alpha=0.2, linestyle='--', linewidth=0.5)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)

# Set tick label sizes
ax3.tick_params(axis='y', which='major', labelsize=14)

# REPOSITIONED LEGEND - moved to top center to avoid blocking bars and values
ax3.legend(loc='upper center', frameon=False, fancybox=True, 
          shadow=False, fontsize=11, ncol=2, borderpad=0.4,
          bbox_to_anchor=(0.52, 0.99))  # Position at top center, 2 columns

# Alternative options if you prefer:
# Option 1: Outside the plot area on top
# ax3.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), 
#           frameon=True, fancybox=True, shadow=False, fontsize=11, ncol=4)

# Option 2: Upper left corner (less likely to block)
# ax3.legend(loc='upper left', frameon=True, fancybox=True, 
#           shadow=False, fontsize=11, ncol=1, borderpad=0.4)

# Option 3: Outside on the right
# ax3.legend(loc='center left', bbox_to_anchor=(1.02, 0.5),
#           frameon=True, fancybox=True, shadow=False, fontsize=11)

# Add panel labels with consistent positioning
fig.text(0.085, 0.91, 'a', fontsize=22, fontweight='bold')
fig.text(0.38, 0.91, 'b', fontsize=22, fontweight='bold')
fig.text(0.662, 0.91, 'c', fontsize=22, fontweight='bold')

# Adjust layout to prevent overlap
plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space for panel labels

plt.savefig('distribution_shift_performance.pdf', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.show()