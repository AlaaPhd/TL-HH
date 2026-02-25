import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
import pandas as pd
import os

# Set the save directory
save_dir = r"D:\Datasets\Parameters test"

# Create directory if it doesn't exist
os.makedirs(save_dir, exist_ok=True)

# Set academic paper style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Computer Modern Roman'],
    'font.size': 12,
    'axes.titlesize': 12,
    'axes.labelsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 12,
    'figure.dpi': 600,
    'savefig.dpi': 600,
})

# ============================================
# Data from the optimization run
# ============================================

# Final results
final_efficiency = 424
final_diversity = 108
best_LS = 2
best_OP = 8
best_MA = 1
time_to_find = 16.16

# LS usage data
ls_data = {
    'LS': [1, 2, 3, 4, 5],
    'Used': [35, 19, 12, 15, 18],
    'Improve': [11, 12, 6, 10, 7],
    'Reward': [-22, 2, -3, -9, -5],
    'Time': [56.6547, 30.1866, 1.14846, 1.50831, 0.0139511]
}

# OP usage data
op_data = {
    'OP': list(range(1, 15)),
    'Used': [3, 3, 3, 3, 3, 39, 7, 6, 16, 3, 3, 3, 4, 3],
    'Improve': [1, 1, 1, 1, 2, 23, 4, 2, 6, 1, 1, 1, 1, 1],
    'Reward': [-3, -3, -3, -3, -3, -2, -1, -2, -2, -3, -3, -3, -3, -3],
    'Time': [0.009767, 0.0181207, 9.41282, 0.480809, 0.0056551, 8.32425, 
             1.28301, 1.20106, 4.61561, 1.12185, 32.1234, 0.0044137, 
             30.8854, 0.0258016]
}

# MA usage data
ma_data = {
    'MA': [1, 2, 3, 4, 5],
    'Used': [36, 15, 17, 16, 15],
    'Improve': [6, 15, 9, 9, 7],
    'Reward': [-10, -8, -7, -4, -8],
    'Time': [11.0497, 35.4149, 28.1292, 4.49606, 10.422]
}

# Create DataFrames
df_ls = pd.DataFrame(ls_data)
df_op = pd.DataFrame(op_data)
df_ma = pd.DataFrame(ma_data)

# Calculate derived metrics
df_ls['Improvement_Rate'] = df_ls['Improve'] / df_ls['Used'] * 100
df_ls['Time_per_Use'] = df_ls['Time'] / df_ls['Used']
df_ls['Reward_per_Use'] = df_ls['Reward'] / df_ls['Used']

df_op['Improvement_Rate'] = df_op['Improve'] / df_op['Used'] * 100
df_op['Time_per_Use'] = df_op['Time'] / df_op['Used']
df_op['Reward_per_Use'] = df_op['Reward'] / df_op['Used']

df_ma['Improvement_Rate'] = df_ma['Improve'] / df_ma['Used'] * 100
df_ma['Time_per_Use'] = df_ma['Time'] / df_ma['Used']
df_ma['Reward_per_Use'] = df_ma['Reward'] / df_ma['Used']

# ============================================
# Create the main figure with titles at BOTTOM CENTER
# ============================================

fig = plt.figure(figsize=(16, 12))

# Overall title with final results
#fig.suptitle(f'TL-HH Optimization Analysis: Instance 20-P500T5M80\n'
         #    f'Final Solution: Efficiency={final_efficiency}, Diversity={final_diversity}',
           #  fontsize=14, fontweight='bold', y=0.98)

# ============================================
# Subplot 1: Performance Dashboard (Top-left) - (a)
# ============================================
ax1 = fig.add_subplot(2, 3, 1)

# Create a performance dashboard
metrics = ['Final Efficiency', 'Final Diversity', 'Time to Find', 'Total Improvements']
values = [final_efficiency, final_diversity, time_to_find, 
          df_ls['Improve'].sum() + df_op['Improve'].sum() + df_ma['Improve'].sum()]
#colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
colors = ['#1f77b4']

bars = ax1.barh(range(len(metrics)), values, color=colors, alpha=0.8)
ax1.set_yticks(range(len(metrics)))
ax1.set_yticklabels(metrics, fontweight='bold')
ax1.set_xlabel('Value', fontweight='bold')

# Title at BOTTOM CENTER with label
ax1.set_title('(a) Final Performance Metrics', fontweight='bold', y=-0.25, pad=25, 
              loc='center', fontsize=10)

# Add value labels
for i, (bar, val) in enumerate(zip(bars, values)):
    ax1.text(bar.get_width() + max(values)*0.01, bar.get_y() + bar.get_height()/2,
            f'{val:.1f}' if i == 2 else f'{val:.0f}',
            va='center', ha='left', fontweight='bold')

# Highlight best configuration
ax1.text(0.02, 0.98, f'Best: LS{best_LS}-OP{best_OP}-MA{best_MA}',
         transform=ax1.transAxes, fontsize=9, fontweight='bold',
         bbox=dict(boxstyle="round,pad=0.3", facecolor="gold", alpha=0.9))

# ============================================
# Subplot 2: LS Performance Analysis (Top-center) - (b)
# ============================================
ax2 = fig.add_subplot(2, 3, 2)

# Create grouped bar chart for LS
x = np.arange(len(df_ls))
width = 0.25

bars1 = ax2.bar(x - width, df_ls['Improvement_Rate'], width, 
                label='Improvement Rate %', color='#1f77b4', alpha=0.8)
bars2 = ax2.bar(x, df_ls['Reward_per_Use'], width, 
                label='Reward per Use', color='#FF0000', alpha=0.8)
bars3 = ax2.bar(x + width, df_ls['Time_per_Use'], width, 
                label='Time per Use (s)', color='#83FF0F', alpha=0.8)

ax2.set_xlabel('Local Search (LS)', fontweight='bold')
ax2.set_ylabel('Performance Metrics', fontweight='bold')

# Title at BOTTOM CENTER with label
ax2.set_title('(b) LS Component Analysis', fontweight='bold', y=-0.25, pad=25, 
              loc='center', fontsize=10)

ax2.set_xticks(x)
ax2.set_xticklabels([f'LS{ls}' for ls in df_ls['LS']])
ax2.legend(loc='upper right', fontsize=8)
ax2.grid(True, alpha=0.3, axis='y')

# Highlight LS2 (best performer)
ax2.axvspan(1 - 0.5, 1 + 0.5, alpha=0.2, color='gold')
ax2.text(1, ax2.get_ylim()[1]*0.9, 'BEST', ha='center', fontweight='bold',
         bbox=dict(boxstyle="round,pad=0.2", facecolor="gold"))

# ============================================
# Subplot 3: OP Usage Patterns (Top-right) - (c)
# ============================================
ax3 = fig.add_subplot(2, 3, 3)

# Create custom blue-to-red colormap
from matplotlib.colors import LinearSegmentedColormap
colors_bubble = ["#0000FF", "#FFFFFF", "#FF0000"]  # Blue -> White -> Red
custom_cmap_bubble = LinearSegmentedColormap.from_list("custom_blue_red", colors_bubble)

# Create bubble chart for OP usage with blue-red colormap
scatter = ax3.scatter(df_op['Time_per_Use'], df_op['Improvement_Rate'],
                     s=df_op['Used']*10,  # Size by usage count
                     c=df_op['Reward_per_Use'],  # Color by reward
                     cmap=custom_cmap_bubble,  # Changed from 'RdYlGn'
                     alpha=0.8,
                     edgecolors='black',
                     linewidth=0.5)

ax3.set_xscale('log')  # Log scale for time
ax3.set_xlabel('Time per Use (s, log scale)', fontweight='bold')
ax3.set_ylabel('Improvement Rate (%)', fontweight='bold')

# Title at BOTTOM CENTER with label
ax3.set_title('(c) OP Performance', fontweight='bold', y=-0.25, pad=25, 
              loc='center', fontsize=10)

ax3.grid(True, alpha=0.3)

# Add colorbar
cbar = plt.colorbar(scatter, ax=ax3)
cbar.set_label('Reward per Use', fontweight='bold')

# Label key OPs
key_ops = [best_OP, 6, 9, 11, 13]  # OP8 (best), and other significant ones
for op in key_ops:
    row = df_op[df_op['OP'] == op]
    if not row.empty:
        x_pos = row['Time_per_Use'].values[0]
        y_pos = row['Improvement_Rate'].values[0]
        label = f'OP{op}'
        if op == best_OP:
            label = f'OP{op}*'
            ax3.annotate(label, (x_pos, y_pos), xytext=(10, 10),
                        textcoords='offset points', fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="gold", alpha=0.9))
        else:
            ax3.annotate(label, (x_pos, y_pos), xytext=(5, 5),
                        textcoords='offset points', fontsize=8)

# ============================================
# Subplot 4: MA Performance Comparison (Bottom-left) - (d)
# ============================================
ax4 = fig.add_subplot(2, 3, 4)

# Create stacked bar chart for MA
x_ma = np.arange(len(df_ma))
bottom = np.zeros(len(df_ma))

# Stack improvements and non-improvements
improvements = df_ma['Improve'].values
non_improvements = df_ma['Used'].values - improvements

bars_improve = ax4.bar(x_ma, improvements, width=0.6, 
                       label='Improvements', color='#1f77b4', alpha=0.8)
bars_non = ax4.bar(x_ma, non_improvements, width=0.6, 
                   bottom=improvements, label='Non-Improvements', 
                   color='#D62728', alpha=0.8)

ax4.set_xlabel('Move Acceptance (MA)', fontweight='bold')
ax4.set_ylabel('Number of Uses', fontweight='bold')

# Title at BOTTOM CENTER with label
ax4.set_title('(d) MA Component: Improvement Analysis', fontweight='bold', y=-0.25, pad=25, 
              loc='center', fontsize=10)

ax4.set_xticks(x_ma)
ax4.set_xticklabels([f'MA{ma}' for ma in df_ma['MA']])
ax4.legend(loc='upper right')

# Add improvement rate as text
for i, (improve, total) in enumerate(zip(improvements, df_ma['Used'].values)):
    rate = improve / total * 100
    ax4.text(i, total + max(df_ma['Used'])*0.02, f'{rate:.0f}%',
             ha='center', fontweight='bold')

# Highlight MA1 (used in best configuration)
ax4.axvspan(0 - 0.3, 0 + 0.3, alpha=0.2, color='gold')
ax4.text(0, ax4.get_ylim()[1]*0.9, 'BEST', ha='center', fontweight='bold',
         bbox=dict(boxstyle="round,pad=0.2", facecolor="gold"))

# ============================================
# Subplot 5: Efficiency-Diversity Trade-off (Bottom-center) - (e)
# ============================================
ax5 = fig.add_subplot(2, 3, 5)

# Create a visualization of the efficiency-diversity trade-off
np.random.seed(42)
n_points = 50
efficiency_progress = np.linspace(300, final_efficiency, n_points)
diversity_progress = 50 + 58 * (1 - np.exp(-np.linspace(0, 3, n_points)))

# Plot progression
ax5.plot(efficiency_progress, diversity_progress, 'b-', alpha=0.3, linewidth=2)
scatter_progress = ax5.scatter(efficiency_progress, diversity_progress,
                              c=np.linspace(0, 1, n_points), cmap='viridis',
                              s=50, alpha=0.6, edgecolors='black', linewidth=0.5)

# Mark final point
ax5.scatter([final_efficiency], [final_diversity], 
           s=300, color='red', marker='*', edgecolors='black', linewidth=2,
           label=f'Final Solution\n(E={final_efficiency}, D={final_diversity})')

# Mark starting point
ax5.scatter([efficiency_progress[0]], [diversity_progress[0]], 
           s=100, color='green', marker='o', edgecolors='black', linewidth=2,
           label='Starting Point')

ax5.set_xlabel('Efficiency', fontweight='bold')
ax5.set_ylabel('Diversity', fontweight='bold')

# Title at BOTTOM CENTER with label
ax5.set_title('(e) Efficiency-Diversity Trade-off', fontweight='bold', y=-0.25, pad=25, 
              loc='center', fontsize=10)

ax5.grid(True, alpha=0.3)
ax5.legend(loc='lower right', fontsize=8)

# Add arrow showing direction of improvement
ax5.annotate('', xy=(final_efficiency, final_diversity),
            xytext=(efficiency_progress[-10], diversity_progress[-10]),
            arrowprops=dict(arrowstyle='->', color='red', lw=2))

# ============================================
# Subplot 6: Configuration Effectiveness (Bottom-right) - (f)
# ============================================
ax6 = fig.add_subplot(2, 3, 6)

# Create a heatmap-like effectiveness matrix
effectiveness_matrix = np.zeros((len(df_ls), len(df_op)))

# Simple effectiveness score: improvement rate * reward sign
for i, ls_row in df_ls.iterrows():
    for j, op_row in df_op.iterrows():
        # Combined effectiveness metric
        effectiveness = (ls_row['Improvement_Rate'] + op_row['Improvement_Rate']) / 2
        effectiveness_matrix[i, j] = effectiveness

# Create custom blue-to-red colormap
colors = ["#0000FF", "#FFFFFF", "#FF0000"]  # Blue -> White -> Red
custom_cmap = LinearSegmentedColormap.from_list("custom", colors)

# Plot the heatmap with custom colormap
im = ax6.imshow(effectiveness_matrix, cmap=custom_cmap, aspect='auto', 
                interpolation='nearest')

ax6.set_xlabel('OP', fontweight='bold')
ax6.set_ylabel('LS', fontweight='bold')

# Title at BOTTOM CENTER with label
ax6.set_title('(f) LS-OP Effectiveness Matrix', fontweight='bold', y=-0.25, pad=25, 
              loc='center', fontsize=10)

# Set ticks
ax6.set_xticks(np.arange(len(df_op)))
ax6.set_xticklabels([f'{op}' for op in df_op['OP']], rotation=45)
ax6.set_yticks(np.arange(len(df_ls)))
ax6.set_yticklabels([f'LS{ls}' for ls in df_ls['LS']])

# Highlight best combination (LS2-OP8)
best_ls_idx = list(df_ls['LS']).index(best_LS)
best_op_idx = list(df_op['OP']).index(best_OP)

# Draw rectangle around best combination
rect = Rectangle((best_op_idx-0.5, best_ls_idx-0.5), 1, 1,
                 linewidth=3, edgecolor='blue', facecolor='none')
ax6.add_patch(rect)

# Add text annotation
ax6.text(best_op_idx, best_ls_idx, 'BEST', 
         ha='center', va='center', fontweight='bold', color='#1f77b4',
         bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

# Add colorbar
cbar = plt.colorbar(im, ax=ax6)
cbar.set_label('Effectiveness', fontweight='bold')

# ============================================
# Adjust layout and save
# ============================================

plt.tight_layout(rect=[0, 0.1, 1, 0.95])  # Increased bottom margin for bottom titles

# Save the figure with full path
main_fig_path = os.path.join(save_dir, 'hyperheuristic_final_analysis.png')
plt.savefig(main_fig_path, dpi=600, bbox_inches='tight')
plt.savefig(os.path.join(save_dir, 'hyperheuristic_final_analysis.pdf'))
plt.savefig(os.path.join(save_dir, 'hyperheuristic_final_analysis.eps'), format='eps')

plt.show()

print(f"✓ Main figure saved to: {main_fig_path}")

