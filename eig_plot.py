import numpy as np
import matplotlib.pyplot as plt

# SB model files and labels
sb_files = [
        ('sb_model/pred_data/eig_pred_sb_asym_normal.npz', 'SB: PUNN'),
        ('sb_model/pred_data/eig_nc_pred_sb_asym_coeff.npz', 'SB: $\mathfrak{su}$(n)-PUNN'),
        ('sb_model/pred_data/eig_pred_sb_asym_pinn.npz', 'SB: PINN'),
        ('sb_model/pred_data/eig_fc_pred_sb_asym_coeff.npz', 'SB: $\mathfrak{su}$(n)-PINN')
]

# FMO complex files and labels
fmo_files = [
        ('fmo_model/pred_data/eig_pred_fmo_7_1_normal.npz', 'FMO: PUNN'),
        ('fmo_model/pred_data/eig_nc_pred_fmo_7_1_coeff.npz', 'FMO: $\mathfrak{su}$(n)-PUNN'),
        ('fmo_model/pred_data/eig_pred_fmo_7_1_pinn.npz', 'FMO: PINN'),
        ('fmo_model/pred_data/eig_fc_pred_fmo_7_1_coeff.npz', 'FMO: $\mathfrak{su}$(n)-PINN')
]

# Create a figure for the 3x2 layout
fig, axes = plt.subplots(4, 2, figsize=(6, 7), sharex='col', constrained_layout=True)

# Helper function for plotting eigenvalues
def plot_eigenvalues(ax, file, label, is_sb=True, legend=False):
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
    ax.set_title(f'{label}', fontsize=12)
    ax.grid(alpha=0.5)

    # Add legend only if specified
    if legend:
        ax.legend(fontsize=8, loc="center", bbox_to_anchor=(0.6, 0.5), ncol=1, frameon=True)
    # Annotate subplot with the negative eigenvalue count
    ax.text(0.9, 0.85, f'Negative eigen values: {negative_count}', transform=ax.transAxes,
            fontsize=10, color='red', ha='right', va='center',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

# Plot for SB model (1st column)
for idx, (ax, (file, label)) in enumerate(zip(axes[:, 0], sb_files)):
    plot_eigenvalues(ax, file, label, is_sb=True, legend=(idx == 0))

# Plot for FMO complex (2nd column)
for idx, (ax, (file, label)) in enumerate(zip(axes[:, 1], fmo_files)):
    plot_eigenvalues(ax, file, label, is_sb=False, legend=(idx == 0))

# Set shared X and Y labels
axes[-1, 0].set_xlabel('Time (1/Δ)', fontsize=12)  # SB model
axes[-1, 1].set_xlabel('Time (ps)', fontsize=12)  # FMO complex
fig.supylabel('Eigenvalues', fontsize=12)

axes[0, 0].text(-0.20, 1.2, 'A', transform=axes[0, 0].transAxes, fontsize=14, fontweight='bold', va='top', ha='left')
axes[1, 0].text(-0.20, 1.2, 'B', transform=axes[1, 0].transAxes, fontsize=14, fontweight='bold', va='top', ha='left')
axes[2, 0].text(-0.20, 1.2, 'C', transform=axes[2, 0].transAxes, fontsize=14, fontweight='bold', va='top', ha='left')
axes[3, 0].text(-0.20, 1.2, 'D', transform=axes[3, 0].transAxes, fontsize=14, fontweight='bold', va='top', ha='left')

axes[0, 1].text(-0.20, 1.2, 'E', transform=axes[0, 1].transAxes, fontsize=14, fontweight='bold', va='top', ha='left')
axes[1, 1].text(-0.20, 1.2, 'F', transform=axes[1, 1].transAxes, fontsize=14, fontweight='bold', va='top', ha='left')
axes[2, 1].text(-0.20, 1.2, 'G', transform=axes[2, 1].transAxes, fontsize=14, fontweight='bold', va='top', ha='left')
axes[3, 1].text(-0.20, 1.2, 'H', transform=axes[3, 1].transAxes, fontsize=14, fontweight='bold', va='top', ha='left')

# Adjust layout and save the figure
plt.subplots_adjust(hspace=0.3)
plt.savefig('sb_fmo_eig.pdf', format='pdf', dpi=300, bbox_inches='tight')
plt.show()

