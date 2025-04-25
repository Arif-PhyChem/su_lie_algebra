import numpy as np
import matplotlib.pyplot as plt

# File names and their labels
dyn_files = [
    ('pred_data/dyn_mlp_sb_asym_punn.npz', 'MLP'),
    ('pred_data/dyn_lstm_sb_asym_punn.npz', 'LSTM'),
    ('pred_data/dyn_cnn_sb_asym_punn.npz', 'CNN'),
    ('../pred_data/pred_sb_asym_normal.npz', 'CNN-LSTM')
]


# Reference
ex = np.load('/home/dell/arif/pypackage/sb/data/test_set/2_epsilon-1.0_Delta-1.0_lambda-0.6_gamma-9.0_beta-1.0.npy')
# Create a figure for subplots
fig, axes = plt.subplots(2, 2, figsize=(6, 6), sharex=True, constrained_layout=True)
# Loop through files and plot
def plot_dyn(ax, label, file, legend):
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
    ax.plot(time_pred, re_rho_11, color=colors[0], linewidth=1.5,   label="Re($\\rho_{11}$)" if legend  else "")
    ax.plot(time_pred, re_rho_22, color=colors[1], linewidth=1.5,  label="Re($\\rho_{22}$)" if legend  else "")
    ax.plot(time_pred, re_rho_12, color=colors[2], linewidth=1.5,    label="Re($\\rho_{12}$)" if legend  else "")
    ax.plot(time_pred, im_rho_12, color=colors[3], linewidth=1.5, label="Im($\\rho_{12}$)" if legend  else "")

    ax.plot(time_ref, ref_re_rho_11,  lw=2.0, ls='None', marker='o', markevery=25, markersize=5, color=colors[0])
    ax.plot(time_ref, ref_re_rho_22,  lw=2.0, ls='None', marker='o', markevery=25, markersize=5, color=colors[1])
    ax.plot(time_ref, ref_re_rho_12,  lw=2.0, ls='None', marker='o', markevery=25, markersize=5, color=colors[2])
    ax.plot(time_ref, ref_im_rho_12,  lw=2.0, ls='None', marker='o', markevery=25, markersize=5, color=colors[3])

    # Customize subplot
    ax.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.7)  # Reference line
    ax.text(8.0, 1.1, label, fontsize=14)
    ax.grid(alpha=0.5)
    if legend:
        ax.legend(fontsize=12, loc="center", bbox_to_anchor=(0.75, 0.7), ncol=1, frameon=True)


for file, label in dyn_files:
    if label == 'MLP':
        ax = axes[0, 0]
    if label == 'LSTM':
        ax = axes[0, 1]
    if label == 'CNN':
        ax = axes[1, 0]
    if label == 'CNN-LSTM':
        ax = axes[1, 1]

    plot_dyn(ax, label, file, legend=(label=='MLP'))


axes[1, 0].set_xlabel('Time (1/Δ)', fontsize=14)
axes[1, 1].set_xlabel('Time (1/Δ)', fontsize=14)


# Adjust layout
fig.tight_layout(pad=1.0)
plt.subplots_adjust(hspace=0.2)  # Space between subplots

axes[0, 0].text(-0.10, 1.10, 'A', transform=axes[0, 0].transAxes, fontsize=14, fontweight='bold', va='top', ha='left')
axes[0, 1].text(-0.20, 1.10, 'B', transform=axes[0, 1].transAxes, fontsize=14, fontweight='bold', va='top', ha='left')
axes[1, 0].text(-0.10, 1.10, 'C', transform=axes[1, 0].transAxes, fontsize=14, fontweight='bold', va='top', ha='left')
axes[1, 1].text(-0.20, 1.10, 'D', transform=axes[1, 1].transAxes, fontsize=14, fontweight='bold', va='top', ha='left')



# Shared X and Y labels
fig.supxlabel('Time($\Delta$)', fontsize=14)
#fig.supylabel("Exciton dynamics", fontsize=14)

# Adjust layout
fig.tight_layout(pad=1.0)
plt.subplots_adjust(hspace=0.2)  # Space between subplots

# Save the figure
plt.savefig("compare_nn_sb_dyn.pdf", format="pdf", dpi=300, bbox_inches="tight")
plt.show()

