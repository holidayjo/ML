import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Image size
img_size = 320

# Anchor boxes grouped by rows
anchor_boxes = [
    [49, 90, 84, 70, 40, 164],
    [59, 119, 60, 160, 68, 177],
    [80, 186, 122, 134, 101, 170]
]

# Normalize the anchor box dimensions to a scale of 0~1
normalized_anchor_boxes = [
    [w / img_size for w in row]
    for row in anchor_boxes
]

# Create a new figure for normalized visualization
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)  # Ensure the y-axis starts from 0 and ends at 1
ax.set_aspect('equal')
ax.set_title("Normalized Anchor Boxes Visualization")
ax.set_xlabel("Normalized Width")
ax.set_ylabel("Normalized Height")

# Plot each normalized anchor box as a rectangle
colors = ['red', 'blue', 'green']
labels = ['40×40', '20×20', '10×10']  # Updated labels
for i, row in enumerate(normalized_anchor_boxes):
    for j in range(0, len(row), 2):
        width = row[j]
        height = row[j + 1]
        # Center the normalized anchor boxes at (0.5, 0.5)
        x_center = 0.5
        y_center = 0.5
        x = x_center - width / 2
        y = y_center - height / 2

        # Add rectangle
        rect = patches.Rectangle((x, y), width, height, linewidth=2, edgecolor=colors[i], facecolor='none', label=labels[i] if j == 0 else None)
        ax.add_patch(rect)

# Add legend with increased font size
handles, labels = ax.get_legend_handles_labels()
unique_handles = []
unique_labels = []
for handle, label in zip(handles, labels):
    if label not in unique_labels:
        unique_handles.append(handle)
        unique_labels.append(label)
ax.legend(unique_handles, unique_labels, loc='upper right', fontsize=12)  # Adjust fontsize as needed

# Show the plot
plt.grid(True)
plt.show()
