import numpy as np
import matplotlib.pyplot as plt

# File names and their labels
# File names and their labels
trace_files = [
    ('../pred_data/trace_pred_fmo_7_1_normal.npz', 'PUNN'),
    ('pred_data/trace_pred_fmo_7_1_normal.npz', 'TC-PUNN')
]
# File names and their labels
eig_files = [
    ('../pred_data/eig_pred_fmo_7_1_normal.npz', 'PUNN'),
    ('pred_data/eig_pred_fmo_7_1_normal.npz', 'TC-PUNN'),
]
# File names and their labels
dyn_files = [
    ('../pred_data/pred_fmo_7_1_normal.npz', 'PUNN'),
    ('pred_data/pred_fmo_7_1_normal.npz', 'TC-PUNN')
]

# Reference trajectory
ref_data = np.load('/home/dell/arif/pypackage/fmo/7_sites_adolph_renger_H/test_data/init_1/7_initial-1_gamma-400.0_lambda-40.0_temp-90.0.npy')
ref_time = ref_data[:, 0]  # Time of reference trajectory
ref_indices = [1, 9, 17, 25, 33, 41, 49]  # Indices for diagonal elements

# Create a figure for subplots
fig, axes = plt.subplots(4, 2, figsize=(6, 8), sharex='col', sharey='row', constrained_layout=True)

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

# Loop through dynamics files and plot
for idx, (ax, (file, label)) in enumerate(zip(axes[2, :], dyn_files)):
    # Load predicted density matrix data
    data = np.load(file)
    time_pred = data['time']  # Time array
    rho = data['rho']         # Shape: [401, 7, 7]

    # Extract real diagonal elements (Re(rho_ii))
    re_rho_diag = [np.real(rho[:, i, i]) for i in range(7)]

    # Extract corresponding reference diagonal elements
    ref_re_rho_diag = [ref_data[:, index] for index in ref_indices]

    # Plot diagonal elements
    colors = ['blue', 'green', 'red', 'orange', 'purple', 'brown', 'cyan']  # Distinct colors for each site
    for i in range(7):
        ax.plot(time_pred, re_rho_diag[i], color=colors[i], linewidth=1.5, label=f"$\\rho_{{{i+1}{i+1}}}$" if idx == 0 else "")
        ax.plot(ref_time/1000, ref_re_rho_diag[i], lw=2.0, ls='None', marker='o', markevery=25, markersize=5, color=colors[i])

    # Customize subplot
    ax.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.7)  # Reference line
    ax.set_title(label, fontsize=14)
    ax.set_xlim(0, 1)
    ax.grid(alpha=0.5)

    # Add legend to the first subplot
    if idx == 0:
        ax.legend(fontsize=10, loc="center", bbox_to_anchor=(0.85, 0.5), ncol=1, frameon=True)

###############################
#   Off diagonal elements
###############################


ref_indices = [2, 10, 18]  # Indices for rho_12, rho_23, rho_34

# Loop through density files and plot
for idx, (ax, (file, label)) in enumerate(zip(axes[3,:], dyn_files)):
    # Load predicted density matrix data
    data = np.load(file)
    time_pred = data['time']  # Time array
    rho = data['rho']         # Shape: [201, 7, 7]

    # Extract real and imaginary parts of off-diagonal elements
    re_rho_12 = np.real(rho[:, 0, 1])  # Re(rho_12)
    im_rho_12 = np.imag(rho[:, 0, 1])  # Im(rho_12)
    re_rho_23 = np.real(rho[:, 1, 2])  # Re(rho_23)
    im_rho_23 = np.imag(rho[:, 1, 2])  # Im(rho_23)
    re_rho_34 = np.real(rho[:, 2, 3])  # Re(rho_34)
    im_rho_34 = np.imag(rho[:, 2, 3])  # Im(rho_34)

    # Extract corresponding reference off-diagonal elements
    ref_re_rho = [ref_data[:, index] for index in ref_indices]  # Reference real parts
    ref_im_rho = [np.imag(ref_data[:, index]) for index in ref_indices]  # Reference imaginary parts

    # Plot real and imaginary parts
    colors_real = ['blue', 'green', 'red']   # Colors for real parts
    colors_imag = ['cyan', 'orange', 'purple']  # Colors for imaginary parts
    labels_real = [r"Re($\rho_{12}$)", r"Re($\rho_{23}$)", r"Re($\rho_{34}$)"]
    labels_imag = [r"Im($\rho_{12}$)", r"Im($\rho_{23}$)", r"Im($\rho_{34}$)"]

    # Plot predicted values
    for i, (re_rho, im_rho, color_re, color_im, label_re, label_im) in enumerate(
        zip(
            [re_rho_12, re_rho_23, re_rho_34],
            [im_rho_12, im_rho_23, im_rho_34],
            colors_real,
            colors_imag,
            labels_real,
            labels_imag,
        )
    ):
        ax.plot(time_pred, re_rho, color=color_re, linewidth=1.5, label=label_re if idx == 0 else "")
        ax.plot(time_pred, im_rho, color=color_im, linewidth=1.5, label=label_im if idx == 0 else "")

    # Plot reference values
    for i, (ref_re, ref_im, color_re, color_im) in enumerate(
        zip(ref_re_rho, ref_im_rho, colors_real, colors_imag)
    ):
        ax.plot(ref_time/1000, ref_re, lw=2.0, ls='None', marker='o', markevery=15, markersize=5, color=color_re)
        ax.plot(ref_time/1000, ref_im, lw=2.0, ls='None', marker='o', markevery=15, markersize=5, color=color_im)

    # Customize subplot
    ax.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.7)  # Reference line
    ax.set_title(label, fontsize=14)
    ax.set_xlim(0, 1)
    ax.grid(alpha=0.5)

    # Add legend to the first subplot
    if idx == 0:
        ax.legend(fontsize=10, loc="center", bbox_to_anchor=(0.75, 0.5), ncol=1, frameon=True)

axes[3, 0].set_xlabel('Time(ps)', fontsize=12)
axes[3, 1].set_xlabel('Time(ps)', fontsize=12)


axes[0, 0].text(-0.30, 1.10, 'A', transform=axes[0, 0].transAxes, fontsize=14, fontweight='bold', va='top', ha='left')
axes[1, 0].text(-0.30, 1.10, 'B', transform=axes[1, 0].transAxes, fontsize=14, fontweight='bold', va='top', ha='left')
axes[2, 0].text(-0.30, 1.10, 'C', transform=axes[2, 0].transAxes, fontsize=14, fontweight='bold', va='top', ha='left')
axes[3, 0].text(-0.30, 1.10, 'D', transform=axes[3, 0].transAxes, fontsize=14, fontweight='bold', va='top', ha='left')

axes[0, 1].text(-0.10, 1.10, 'E', transform=axes[0, 1].transAxes, fontsize=14, fontweight='bold', va='top', ha='left')
axes[1, 1].text(-0.10, 1.10, 'F', transform=axes[1, 1].transAxes, fontsize=14, fontweight='bold', va='top', ha='left')
axes[2, 1].text(-0.10, 1.10, 'G', transform=axes[2, 1].transAxes, fontsize=14, fontweight='bold', va='top', ha='left')
axes[3, 1].text(-0.10, 1.10, 'H', transform=axes[3, 1].transAxes, fontsize=14, fontweight='bold', va='top', ha='left')

# Adjust layout
fig.tight_layout(pad=1.0)
plt.subplots_adjust(hspace=0.2)  # Space between subplots
plt.subplots_adjust(wspace=0.2)  # Space between subplots

# Save and show the plot
plt.savefig('fmo_all.pdf', format='pdf', dpi=300)
plt.show()

