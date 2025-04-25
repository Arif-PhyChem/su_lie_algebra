import numpy as np
import matplotlib.pyplot as plt

# File names and their labels
eig_files = [
    ('pred_data/eig_mlp_sb_asym_punn.npz', 'MLP'),
    ('pred_data/eig_lstm_sb_asym_punn.npz', 'LSTM'),
    ('pred_data/eig_cnn_sb_asym_punn.npz', 'CNN'),
    ('../pred_data/eig_pred_sb_asym_normal.npz', 'CNN-LSTM')
]

# Create a figure for the 3x2 layout
fig, axes = plt.subplots(2, 2, figsize=(6, 6), sharex='col', constrained_layout=True)

# Helper function for plotting eigenvalues
def plot_eigenvalues(ax, label, file, legend):
    # Load data
    data = np.load(file)
    time = data['time']
    eig = data['eig']

    # Separate positive and negative eigenvalues
    eig_positive = np.where(np.real(eig) > 0, np.real(eig), np.nan)
    eig_negative = np.where(np.real(eig) < 0, np.real(eig), np.nan)
    # Count negative eigenvalues
    negative_count = np.sum(np.real(eig) < 0)

    indices = np.arange(0, eig.shape[0], 10)
    # Plot eigenvalues
    for i in range(eig.shape[1]):
        ax.scatter(time[indices], eig_positive[indices, i], color='blue', alpha=0.6, s=5, label='Positive eigenvalue' if legend and i == 0 else "")
    for i in range(eig.shape[1]):
        ax.scatter(time, eig_negative[:, i], color='red', alpha=1.0, s=25, label='Negative eigenvalue' if legend and i == 0 else "")

    # Customize subplot
    ax.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.7)
    ax.set_title(f'{label}', fontsize=14)
    ax.grid(alpha=0.5)

    # Add legend only if specified
    if legend:
        ax.legend(fontsize=10, loc="center", bbox_to_anchor=(0.6, 0.5), ncol=1, frameon=True)
    # Annotate subplot with the negative eigenvalue count
    ax.text(0.9, 0.85, f'Negative eigen values: {negative_count}', transform=ax.transAxes,
            fontsize=12, color='red', ha='right', va='center',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

for file, label in eig_files:
    if label == 'MLP':
        ax = axes[0, 0]
    if label == 'LSTM':
        ax = axes[0, 1]
    if label == 'CNN':
        ax = axes[1, 0]
    if label == 'CNN-LSTM':
        ax = axes[1, 1]

    plot_eigenvalues(ax, label, file, legend=(label=='MLP'))


axes[1, 0].set_xlabel('Time (1/Δ)', fontsize=14)
axes[1, 1].set_xlabel('Time (1/Δ)', fontsize=14)


# Adjust layout
fig.tight_layout(pad=1.0)
plt.subplots_adjust(hspace=0.2)  # Space between subplots

axes[0, 0].text(-0.10, 1.10, 'A', transform=axes[0, 0].transAxes, fontsize=14, fontweight='bold', va='top', ha='left')
axes[0, 1].text(-0.20, 1.10, 'B', transform=axes[0, 1].transAxes, fontsize=14, fontweight='bold', va='top', ha='left')
axes[1, 0].text(-0.10, 1.10, 'C', transform=axes[1, 0].transAxes, fontsize=14, fontweight='bold', va='top', ha='left')
axes[1, 1].text(-0.20, 1.10, 'D', transform=axes[1, 1].transAxes, fontsize=14, fontweight='bold', va='top', ha='left')


# Adjust layout and save the figure
plt.savefig('compare_nn_sb_eig.pdf', format='pdf', dpi=300, bbox_inches='tight')
plt.show()

