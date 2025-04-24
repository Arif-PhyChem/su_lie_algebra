import pandas as pd
import matplotlib.pyplot as plt

# Define file paths
sb_loss_files = [
    ('sb_model/pred_data/normal_loss_history_no_constraint.csv', 'sb_model/pred_data/coeff_loss_history_no_constraint.csv'),
    ('sb_model/pred_data/pinn_loss_history_full_constraints.csv', 'sb_model/pred_data/coeff_loss_history_full_constraints.csv')
]
fmo_loss_files = [
    ('fmo_model/pred_data/normal_loss_history_no_constraint.csv', 'fmo_model/pred_data/coeff_loss_history_no_constraint.csv'),
    ('fmo_model/pred_data/pinn_loss_history_full_constraints.csv', 'fmo_model/pred_data/coeff_loss_history_full_constraints.csv')
]

# Create a figure for subplots
fig, axes = plt.subplots(2, 2, figsize=(7, 6.5), sharex=True, constrained_layout=True)

# Function to plot data
def plot_loss(ax, epochs, training_loss, validation_loss, color_train, color_val, marker_train, marker_val, label_prefix, add_legend):
    ax.plot(epochs, training_loss, label=f"{label_prefix} Training Loss", color=color_train, linestyle="-",
            marker=marker_train, markevery=25, markersize=5)
    ax.plot(epochs, validation_loss, label=f"{label_prefix} Validation Loss", color=color_val, linestyle="-",
            marker=marker_val, markevery=25, markersize=5)
    if add_legend:
        ax.legend(fontsize=10, loc="upper right", ncol=1, frameon=True)

# Loop through SB model files and plot
for i, (ax, (file_1, file_2)) in enumerate(zip(axes[0, :], sb_loss_files)):
    data_1 = pd.read_csv(file_1).iloc[:200]
    data_2 = pd.read_csv(file_2).iloc[:200]

    epochs_1 = data_1["Epoch"]
    training_loss_1 = data_1["Training Loss"]
    validation_loss_1 = data_1["Validation Loss"]

    epochs_2 = data_2["Epoch"]
    training_loss_2 = data_2["Training Loss"]
    validation_loss_2 = data_2["Validation Loss"]

    label_prefix_1 = "PUNN" if i == 0 else "PINN"
    label_prefix_2 = "$\\mathfrak{su}$(n)-PUNN" if i == 0 else "$\\mathfrak{su}$(n)-PINN"

    plot_loss(ax, epochs_1, training_loss_1, validation_loss_1, "blue", "orange", "o", "x", label_prefix_1, True)
    plot_loss(ax, epochs_2, training_loss_2, validation_loss_2, "red", "black", "o", "x", label_prefix_2, True)

    ax.set_yscale("log")
    ax.grid(True)
    ax.set_title(f"SB: {label_prefix_1} vs {label_prefix_2}", fontsize=12)
    if i==1:
        ax.set_yticklabels([])

# Loop through FMO model files and plot
for i, (ax, (file_1, file_2)) in enumerate(zip(axes[1, :], fmo_loss_files)):
    data_1 = pd.read_csv(file_1).iloc[:200]
    data_2 = pd.read_csv(file_2).iloc[:200]

    epochs_1 = data_1["Epoch"]
    training_loss_1 = data_1["Training Loss"]
    validation_loss_1 = data_1["Validation Loss"]

    epochs_2 = data_2["Epoch"]
    training_loss_2 = data_2["Training Loss"]
    validation_loss_2 = data_2["Validation Loss"]

    label_prefix_1 = "PUNN" if i == 0 else "PINN"
    label_prefix_2 = "$\\mathfrak{su}$(n)-PUNN" if i == 0 else "$\\mathfrak{su}$(n)-PINN"

    plot_loss(ax, epochs_1, training_loss_1, validation_loss_1, "blue", "orange", "o", "x", label_prefix_1, True)
    plot_loss(ax, epochs_2, training_loss_2, validation_loss_2, "red", "black", "o", "x", label_prefix_2, True)

    ax.set_yscale("log")
    ax.grid(True)
    ax.set_title(f"FMO: {label_prefix_1} vs {label_prefix_2}", fontsize=12)
    if i==1:
        ax.set_yticklabels([])

# Add labels and global title
axes[1, 0].set_xlabel("Epochs", fontsize=12)
axes[1, 1].set_xlabel("Epochs", fontsize=12)
fig.supylabel("Loss", fontsize=14)
#fig.suptitle("Training and Validation Loss Across Models", fontsize=16)

# Add subplot labels
axes[0, 0].text(-0.1, 1.1, 'A', transform=axes[0, 0].transAxes, fontsize=14, fontweight='bold', va='top', ha='left')
axes[1, 0].text(-0.1, 1.1, 'B', transform=axes[1, 0].transAxes, fontsize=14, fontweight='bold', va='top', ha='left')
axes[0, 1].text(-0.1, 1.1, 'C', transform=axes[0, 1].transAxes, fontsize=14, fontweight='bold', va='top', ha='left')
axes[1, 1].text(-0.1, 1.1, 'D', transform=axes[1, 1].transAxes, fontsize=14, fontweight='bold', va='top', ha='left')


# Adjust layout
fig.tight_layout(pad=1.0)
plt.subplots_adjust(hspace=0.2, wspace=0.1)

# Save the plot to a file for paper use
plt.savefig('lr_sb_fmo_plot.pdf', format='pdf', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()

