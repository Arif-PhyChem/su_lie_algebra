import numpy as np
import matplotlib.pyplot as plt

# File names and their labels
trace_files = [
    ('../pred_data/trace_pred_sb_asym_normal.npz', 'PUNN'),
    ('pred_data/trace_pred_sb_asym_normal.npz', 'TC-PUNN')
]

# SB model files and labels
eig_files = [
        ('../pred_data/eig_pred_sb_asym_normal.npz', 'PUNN'),
        ('pred_data/eig_pred_sb_asym_normal.npz', 'TC-PUNN')
]

# File names and their labels
dyn_files = [
    ('../pred_data/pred_sb_asym_normal.npz', 'PUNN'),
    ('pred_data/pred_sb_asym_normal.npz', 'TC-PUNN')
]

# Reference
ex = np.load('/home/dell/arif/pypackage/sb/data/test_set/2_epsilon-1.0_Delta-1.0_lambda-0.6_gamma-9.0_beta-1.0.npy')


# Create a figure for subplots
fig, axes = plt.subplots(3, 2, figsize=(6, 8), sharex='col', sharey='row', constrained_layout=True)

# Loop through files and plot
for ax, (file, label) in zip(axes[0, :], trace_files):
    # Load data
    data = np.load(file)
    time = data['time']
    trace = data['trace']

    # Plot the trace
    ax.plot(time, trace, color='blue', alpha=0.8, linewidth=1.5, label='Trace')

    # Customize subplot
    ax.axhline(1, color='black', linestyle='--', linewidth=0.8, alpha=0.7)  # Reference line at y=0
    ax.set_ylim(0.990, 1.004)
    ax.set_title(f'{label}', fontsize=14)
    ax.grid(alpha=0.5)

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
    ax.set_title(f'{label}', fontsize=14)
    ax.grid(alpha=0.5)

    # Add legend only if specified
    if legend:
        ax.legend(fontsize=8, loc="center", bbox_to_anchor=(0.6, 0.5), ncol=1, frameon=True)
    # Annotate subplot with the negative eigenvalue count
    ax.text(0.9, 0.85, f'Negative eigen values: {negative_count}', transform=ax.transAxes,
            fontsize=10, color='red', ha='right', va='center',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

# Plot for SB model (1st column)
for idx, (ax, (file, label)) in enumerate(zip(axes[1, :], eig_files)):
    plot_eigenvalues(ax, file, label, is_sb=True, legend=(idx == 0))

# Loop through files and plot
for idx, (ax, (file, label)) in enumerate(zip(axes[2,:], dyn_files)):
    # Load data
    data = np.load(file)
    time_pred = data['time']  # Time array
    rho = data['rho']    # Shape: [401, 2, 2]

    # Extract components
    re_rho_11 = np.real(rho[:, 0, 0])  # Re(rho_11)
    re_rho_22 = np.real(rho[:, 1, 1])  # Re(rho_22)
    re_rho_12 = np.real(rho[:, 0, 1])  # Re(rho_12)
    im_rho_12 = np.imag(rho[:, 0, 1])  # Im(rho_12)
    # Extract components of ref traj
    time_ref = np.real(ex[:, 0]) # Time of ref traj
    ref_re_rho_11 = np.real(ex[:, 1])  # Ref Re(rho_11)
    ref_re_rho_22 = np.real(ex[:, 4])  # Ref Re(rho_22)
    ref_re_rho_12 = np.real(ex[:, 2])  # Ref Re(rho_12)
    ref_im_rho_12 = np.imag(ex[:, 2])  # Ref Im(rho_12)

    # Plot each component
    colors = ['blue', 'green', 'red', 'orange', 'purple', 'brown', 'cyan']  # Distinct colors for each site
    ax.plot(time_pred, re_rho_11, color=colors[0], linewidth=1.5,   label="Re($\\rho_{11}$)" if idx == 0  else "")
    ax.plot(time_pred, re_rho_22, color=colors[1], linewidth=1.5,  label="Re($\\rho_{22}$)" if idx == 0  else "")
    ax.plot(time_pred, re_rho_12, color=colors[2], linewidth=1.5,    label="Re($\\rho_{12}$)" if idx == 0  else "")
    ax.plot(time_pred, im_rho_12, color=colors[3], linewidth=1.5, label="Im($\\rho_{12}$)" if idx == 0  else "")

    ax.plot(time_ref, ref_re_rho_11,  lw=2.0, ls='None', marker='o', markevery=25, markersize=5, color=colors[0])
    ax.plot(time_ref, ref_re_rho_22,  lw=2.0, ls='None', marker='o', markevery=25, markersize=5, color=colors[1])
    ax.plot(time_ref, ref_re_rho_12,  lw=2.0, ls='None', marker='o', markevery=25, markersize=5, color=colors[2])
    ax.plot(time_ref, ref_im_rho_12,  lw=2.0, ls='None', marker='o', markevery=25, markersize=5, color=colors[3])

    # Customize subplot
    ax.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.7)  # Reference line
    ax.set_title(f'{label}', fontsize=14)
    ax.grid(alpha=0.5)
    if idx == 0:
        ax.legend(fontsize=12, loc="center", bbox_to_anchor=(0.75, 0.7), ncol=1, frameon=True)

axes[2, 0].set_xlabel('Time($1/\Delta$)', fontsize=12)
axes[2, 1].set_xlabel('Time($1/\Delta$)', fontsize=12)


axes[0, 0].text(-0.20, 1.10, 'A', transform=axes[0, 0].transAxes, fontsize=14, fontweight='bold', va='top', ha='left')
axes[1, 0].text(-0.20, 1.10, 'B', transform=axes[1, 0].transAxes, fontsize=14, fontweight='bold', va='top', ha='left')
axes[2, 0].text(-0.20, 1.10, 'C', transform=axes[2, 0].transAxes, fontsize=14, fontweight='bold', va='top', ha='left')

axes[0, 1].text(-0.10, 1.10, 'D', transform=axes[0, 1].transAxes, fontsize=14, fontweight='bold', va='top', ha='left')
axes[1, 1].text(-0.10, 1.10, 'E', transform=axes[1, 1].transAxes, fontsize=14, fontweight='bold', va='top', ha='left')
axes[2, 1].text(-0.10, 1.10, 'F', transform=axes[2, 1].transAxes, fontsize=14, fontweight='bold', va='top', ha='left')

# Adjust layout
fig.tight_layout(pad=1.0)
plt.subplots_adjust(hspace=0.2)  # Space between subplots
plt.subplots_adjust(wspace=0.2)  # Space between subplots

# Save and show the plot
plt.savefig('sb_all.pdf', format='pdf', dpi=300)
plt.show()

