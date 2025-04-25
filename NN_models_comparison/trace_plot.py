import numpy as np
import matplotlib.pyplot as plt

# File names and their labels
trace_files = [
    ('pred_data/trace_mlp_sb_asym_punn.npz', 'MLP'),
    ('pred_data/trace_lstm_sb_asym_punn.npz', 'LSTM'),
    ('pred_data/trace_cnn_sb_asym_punn.npz', 'CNN'),
    ('../pred_data/trace_pred_sb_asym_normal.npz', 'CNN-LSTM')
]

# Create a figure for subplots
fig, axes = plt.subplots(2, 2, figsize=(6, 6), sharex=True, constrained_layout=True)


def plot_the_trace(ax, label, file):
    # Load data
    data = np.load(file)
    time = data['time']
    trace = data['trace']

    # Plot the trace
    ax.plot(time, trace, color='blue', alpha=0.8, linewidth=1.5, label='Trace')

    # Customize subplot
    ax.axhline(1, color='black', linestyle='--', linewidth=0.8, alpha=0.7)  # Reference line at y=0
    ax.set_ylim(0.990, 1.01)
    ax.set_title(f'{label}', fontsize=14)
    ax.grid(alpha=0.5)


for file, label in trace_files:
    if label == 'MLP':
        ax = axes[0, 0]
    if label == 'LSTM':
        ax = axes[0, 1]
    if label == 'CNN':
        ax = axes[1, 0]
    if label == 'CNN-LSTM':
        ax = axes[1, 1]

    plot_the_trace(ax, label, file)


axes[1, 0].set_xlabel('Time (1/Δ)', fontsize=14)
axes[1, 1].set_xlabel('Time (1/Δ)', fontsize=14)

#fig.supylabel('Trace', fontsize=14)

# Adjust layout
fig.tight_layout(pad=1.0)
plt.subplots_adjust(hspace=0.2)  # Space between subplots

axes[0, 0].text(-0.20, 1.10, 'A', transform=axes[0, 0].transAxes, fontsize=14, fontweight='bold', va='top', ha='left')
axes[0, 1].text(-0.20, 1.10, 'B', transform=axes[0, 1].transAxes, fontsize=14, fontweight='bold', va='top', ha='left')
axes[1, 0].text(-0.20, 1.10, 'C', transform=axes[1, 0].transAxes, fontsize=14, fontweight='bold', va='top', ha='left')
axes[1, 1].text(-0.20, 1.10, 'D', transform=axes[1, 1].transAxes, fontsize=14, fontweight='bold', va='top', ha='left')

# Save the plot to a file for paper use
plt.savefig('compare_nn_sb_trace.pdf', format='pdf', dpi=300, bbox_inches='tight')
plt.show()

