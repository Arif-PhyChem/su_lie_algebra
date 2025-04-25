import numpy as np
import matplotlib.pyplot as plt

# File names and labels
sb_density_files = [
    ('sb_model/pred_data/asym_dynamics_PUNN.npz', 'PUNN'),
    ('sb_model/pred_data/asym_dynamics_su2-PUNN.npz', '$\mathfrak{su}$(n)-PUNN'),
    ('sb_model/pred_data/asym_dynamics_PINN.npz', 'PINN'),
    ('sb_model/pred_data/asym_dynamics_su2-PINN.npz', '$\mathfrak{su}$(n)-PINN')
]
fmo_density_files = [
    ('fmo_complex/pred_data/dynamics_fmo_7_1_PUNN.npz', 'PUNN'),
    ('fmo_complex/pred_data/dynamics_fmo_7_1_su7-PUNN.npz', '$\mathfrak{su}$(n)-PUNN'),
    ('fmo_complex/pred_data/dynamics_fmo_7_1_PINN.npz', 'PINN'),
    ('fmo_complex/pred_data/dynamics_fmo_7_1_su7-PINN.npz', '$\mathfrak{su}$(n)-PINN')
]

# Create a figure with 3 rows and 2 columns
fig, axes = plt.subplots(4, 2, figsize=(6, 7), constrained_layout=True, sharey=True)

# Plot for spin-boson model (left column)
for i, (file, label) in enumerate(sb_trace_files):
    data = np.load(file)
    time = data['time']
    trace = data['trace']
    ax = axes[i, 0]  # Left column

    ax.plot(time, trace, color='blue', alpha=0.8, linewidth=1.5, label='Trace')
    ax.axhline(1, color='black', linestyle='--', linewidth=0.8, alpha=0.7)  # Reference line
    ax.set_ylim(0.99, 1.01)
    ax.set_title(f'SB: {label}', fontsize=12)
    ax.grid(alpha=0.5)
    if i == 3:  # Add x-axis label only for the last row
        ax.set_xlabel('Time (1/Δ)', fontsize=12)

# Plot for FMO complex (right column)
for i, (file, label) in enumerate(fmo_trace_files):
    data = np.load(file)
    time = data['time']
    trace = data['trace']
    ax = axes[i, 1]  # Right column

    ax.plot(time, trace, color='blue', alpha=0.8, linewidth=1.5, label='Trace')
    ax.axhline(1, color='black', linestyle='--', linewidth=0.8, alpha=0.7)  # Reference line
    ax.set_ylim(0.99, 1.01)
    ax.set_title(f'FMO: {label}', fontsize=12)
    ax.grid(alpha=0.5)
    if i == 3:  # Add x-axis label only for the last row
        ax.set_xlabel('Time (ps)', fontsize=12)


axes[0, 0].text(-0.20, 1.3, 'A', transform=axes[0, 0].transAxes, fontsize=14, fontweight='bold', va='top', ha='left')
axes[1, 0].text(-0.20, 1.3, 'B', transform=axes[1, 0].transAxes, fontsize=14, fontweight='bold', va='top', ha='left')
axes[2, 0].text(-0.20, 1.3, 'C', transform=axes[2, 0].transAxes, fontsize=14, fontweight='bold', va='top', ha='left')
axes[3, 0].text(-0.20, 1.3, 'D', transform=axes[3, 0].transAxes, fontsize=14, fontweight='bold', va='top', ha='left')

axes[0, 1].text(-0.20, 1.3, 'E', transform=axes[0, 1].transAxes, fontsize=14, fontweight='bold', va='top', ha='left')
axes[1, 1].text(-0.20, 1.3, 'F', transform=axes[1, 1].transAxes, fontsize=14, fontweight='bold', va='top', ha='left')
axes[2, 1].text(-0.20, 1.3, 'G', transform=axes[2, 1].transAxes, fontsize=14, fontweight='bold', va='top', ha='left')
axes[3, 1].text(-0.20, 1.3, 'H', transform=axes[3, 1].transAxes, fontsize=14, fontweight='bold', va='top', ha='left')


# Add shared y-axis label
fig.supylabel('Trace', fontsize=14)

# Adjust layout
fig.tight_layout(pad=1.0)
plt.subplots_adjust(hspace=0.6, wspace=0.4)

# Save the plot to a file for paper use
plt.savefig('sb_fmo_trace_plot.pdf', format='pdf', dpi=300, bbox_inches='tight')
plt.show()

