import numpy as np
import matplotlib.pyplot as plt

# File names and labels
sb_density_files = [
    ('sb_model/pred_data/pred_sb_asym_normal.npz', 'PUNN'),
    ('sb_model/pred_data/nc_pred_sb_asym_coeff.npz', '$\mathfrak{su}$(n)-PUNN'),
    ('sb_model/pred_data/pred_sb_asym_pinn.npz', 'PINN'),
    ('sb_model/pred_data/fc_pred_sb_asym_coeff.npz', '$\mathfrak{su}$(n)-PINN')
]
fmo_density_files = [
    ('fmo_model/pred_data/pred_fmo_7_1_normal.npz', 'PUNN'),
    ('fmo_model/pred_data/nc_pred_fmo_7_1_coeff.npz', '$\mathfrak{su}$(n)-PUNN'),
    ('fmo_model/pred_data/pred_fmo_7_1_pinn.npz', 'PINN'),
    ('fmo_model/pred_data/fc_pred_fmo_7_1_coeff.npz', '$\mathfrak{su}$(n)-PINN')
]

# Reference data for SB model
sb_ref = np.load('/home/dell/arif/pypackage/sb/data/test_set/2_epsilon-1.0_Delta-1.0_lambda-0.6_gamma-9.0_beta-1.0.npy')
sb_ref_time = np.real(sb_ref[:, 0])  # Reference time

# Reference data for FMO complex
fmo_ref = np.load('/home/dell/arif/pypackage/fmo/7_sites_adolph_renger_H/test_data/init_1/7_initial-1_gamma-400.0_lambda-40.0_temp-90.0.npy')
fmo_ref_time = fmo_ref[:, 0] / 1000  # Convert time to ps
fmo_diag_indices = [1, 9, 17, 25, 33, 41, 49]  # Diagonal element indices
fmo_offdiag_indices = [2, 10, 18]  # Off-diagonal elements: rho_12, rho_23, rho_34

# Create a 3x3 figure
fig, axes = plt.subplots(4, 3, figsize=(8, 10), constrained_layout=True, sharex='col')

# Colors for plots
colors_diag = ['blue', 'green']  # For SB diagonal
colors_offdiag_sb = ['red', 'orange']  # For SB off-diagonal
colors_fmo = ['blue', 'green', 'red', 'orange', 'purple', 'brown', 'cyan']  # For FMO
colors_fmo_re_offdiag = ['blue', 'green', 'red'] # For FMO real off-diagonal elements
colors_fmo_im_offdiag = ['orange', 'purple', 'brown'] # For FMO imag off-diagonal elements

# Plot SB model (First Column)
for row_idx, (file, label) in enumerate(sb_density_files):
    data = np.load(file)
    time_pred = data['time']
    rho = data['rho']

    # Diagonal elements
    re_rho_11 = np.real(rho[:, 0, 0])
    re_rho_22 = np.real(rho[:, 1, 1])
    sb_ref_re_rho_11 = np.real(sb_ref[:, 1])
    sb_ref_re_rho_22 = np.real(sb_ref[:, 4])

    # Off-diagonal elements
    re_rho_12 = np.real(rho[:, 0, 1])
    im_rho_12 = np.imag(rho[:, 0, 1])
    sb_ref_re_rho_12 = np.real(sb_ref[:, 2])
    sb_ref_im_rho_12 = np.imag(sb_ref[:, 2])

    ax = axes[row_idx, 0]
    ax.plot(time_pred, re_rho_11, color=colors_diag[0], label="$\\rho_{\mathrm{S},11}$")
    ax.plot(time_pred, re_rho_22, color=colors_diag[1], label="$\\rho_{\mathrm{S},22}$")
    ax.plot(sb_ref_time, sb_ref_re_rho_11, 'o', color=colors_diag[0], markevery=20, markersize=4)
    ax.plot(sb_ref_time, sb_ref_re_rho_22, 'o', color=colors_diag[1], markevery=20, markersize=4)

    ax.plot(time_pred, re_rho_12, color=colors_offdiag_sb[0], label="Re($\\rho_{\mathrm{S},12}$)")
    ax.plot(time_pred, im_rho_12, color=colors_offdiag_sb[1], label="Im($\\rho_{\mathrm{S},12}$)")
    ax.plot(sb_ref_time, sb_ref_re_rho_12, 'o', color=colors_offdiag_sb[0], markevery=20, markersize=4)
    ax.plot(sb_ref_time, sb_ref_im_rho_12, 'o', color=colors_offdiag_sb[1], markevery=20, markersize=4)

    ax.set_title(f"SB: {label}", fontsize=12)
    ax.set_xlim(0, 20)
    ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
    ax.grid(alpha=0.5)
    # Add legend only to the top plot of the column
    if row_idx == 0:
        ax.legend(fontsize=10, loc="upper right", frameon=True)  # Adjust location as needed
    #else:
        #ax.get_legend().remove()  # Remove legend from all other rows
    # Adjust shared x-axis labels
    if row_idx == 3:
        ax.set_xlabel('Time (1/Δ)', fontsize=12)  # SB model (first column)


# Plot FMO diagonal elements (Second Column)
for row_idx, (file, label) in enumerate(fmo_density_files):
    data = np.load(file)
    time_pred = data['time']
    rho = data['rho']

    # Diagonal elements
    re_rho_diag = [np.real(rho[:, i, i]) for i in range(7)]
    ref_re_rho_diag = [fmo_ref[:, idx] for idx in fmo_diag_indices]

    ax = axes[row_idx, 1]
    for i, (diag, ref_diag, color) in enumerate(zip(re_rho_diag, ref_re_rho_diag, colors_fmo)):
        ax.plot(time_pred, diag, color=color, label=f"$\\rho_{{\\mathrm{{S}},{i+1}{i+1}}}$")
        ax.plot(fmo_ref_time, ref_diag, 'o', color=color, markevery=20, markersize=4)

    ax.set_title(f"FMO: {label}", fontsize=12)
    ax.set_xlim(0, 1)
    ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
    ax.grid(alpha=0.5)
    # Add legend only to the top plot of the column
    if row_idx == 0:
        ax.legend(fontsize=10, loc="upper right", frameon=True)  # Adjust location as needed
    #else:
    #    ax.get_legend().remove()  # Remove legend from all other rows

# Plot FMO off-diagonal elements (Third Column)
for row_idx, (file, label) in enumerate(fmo_density_files):
    data = np.load(file)
    time_pred = data['time']
    rho = data['rho']

    # Off-diagonal elements: rho_12, rho_23, rho_34
    re_rho_offdiag = [np.real(rho[:, 0, 1]), np.real(rho[:, 1, 2]), np.real(rho[:, 2, 3])]
    im_rho_offdiag = [np.imag(rho[:, 0, 1]), np.imag(rho[:, 1, 2]), np.imag(rho[:, 2, 3])]
    ref_re_offdiag = [np.real(fmo_ref[:, idx]) for idx in fmo_offdiag_indices]
    ref_im_offdiag = [np.imag(fmo_ref[:, idx]) for idx in fmo_offdiag_indices]

    ax = axes[row_idx, 2]
    labels_real = [r"Re($\rho_{\mathrm{S},12}$)", r"Re($\rho_{\mathrm{S},23}$)", r"Re($\rho_{\mathrm{S},34}$)"]
    labels_imag = [r"Im($\rho_{\mathrm{S},12}$)", r"Im($\rho_{\mathrm{S},23}$)", r"Im($\rho_{\mathrm{S},34}$)"]

    for i, (re_rho, ref_re, label_re, color) in enumerate(
        zip(re_rho_offdiag, ref_re_offdiag, labels_real, colors_fmo_re_offdiag)
    ):
        ax.plot(time_pred, re_rho, color=color, label=label_re)
        ax.plot(fmo_ref_time, ref_re, 'o', color=color, markevery=15, markersize=4)
    
    for i, (im_rho, ref_im, label_im, color) in enumerate(
        zip(im_rho_offdiag, ref_im_offdiag, labels_imag, colors_fmo_im_offdiag)
    ):
        ax.plot(time_pred, im_rho, color=color, label=label_im)
        ax.plot(fmo_ref_time, ref_im, 'o', color=color, markevery=15, markersize=4)

    ax.set_title(f"FMO: {label}", fontsize=12)
    ax.set_xlim(0, 1)
    ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
    ax.grid(alpha=0.5)
    # Add legend only to the top plot of the column
    if row_idx == 0:
        ax.legend(fontsize=10, loc="upper right", frameon=True)  # Adjust location as needed
    #else:
    #    ax.get_legend().remove()  # Remove legend from all other rows

#

#fig.supylabel('Density Matrix Elements', fontsize=14)

# Adjust labels for each column
# Second and third columns: FMO complex with time in ps
axes[3, 1].set_xlabel('Time (ps)', fontsize=12)
axes[3, 2].set_xlabel('Time (ps)', fontsize=12)


axes[0, 0].text(-0.20, 1.20, 'A', transform=axes[0, 0].transAxes, fontsize=14, fontweight='bold', va='top', ha='left')
axes[1, 0].text(-0.20, 1.20, 'B', transform=axes[1, 0].transAxes, fontsize=14, fontweight='bold', va='top', ha='left')
axes[2, 0].text(-0.20, 1.20, 'C', transform=axes[2, 0].transAxes, fontsize=14, fontweight='bold', va='top', ha='left')
axes[3, 0].text(-0.20, 1.20, 'D', transform=axes[3, 0].transAxes, fontsize=14, fontweight='bold', va='top', ha='left')

axes[0, 1].text(-0.20, 1.20, 'E', transform=axes[0, 1].transAxes, fontsize=14, fontweight='bold', va='top', ha='left')
axes[1, 1].text(-0.20, 1.20, 'F', transform=axes[1, 1].transAxes, fontsize=14, fontweight='bold', va='top', ha='left')
axes[2, 1].text(-0.20, 1.20, 'G', transform=axes[2, 1].transAxes, fontsize=14, fontweight='bold', va='top', ha='left')
axes[3, 1].text(-0.20, 1.20, 'H', transform=axes[3, 1].transAxes, fontsize=14, fontweight='bold', va='top', ha='left')

axes[0, 2].text(-0.20, 1.20, 'I', transform=axes[0, 2].transAxes, fontsize=14, fontweight='bold', va='top', ha='left')
axes[1, 2].text(-0.20, 1.20, 'J', transform=axes[1, 2].transAxes, fontsize=14, fontweight='bold', va='top', ha='left')
axes[2, 2].text(-0.20, 1.20, 'K', transform=axes[2, 2].transAxes, fontsize=14, fontweight='bold', va='top', ha='left')
axes[3, 2].text(-0.20, 1.20, 'L', transform=axes[3, 2].transAxes, fontsize=14, fontweight='bold', va='top', ha='left')


# Save and show the plot
plt.savefig('sb_fmo_dyn.pdf', format='pdf', dpi=300)
plt.show()

