import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os

# Set the folder path
folder_path = "/home/siddharth/Documents/pikachu/imgs/style"

# Load the images from the folder and resize them to a common size
target_size = (200, 200)  # Set the target size for all images
images = []
for filename in os.listdir(folder_path):
    if filename.endswith((".jpg", ".jpeg", ".png")):
        image_path = os.path.join(folder_path, filename)
        img = Image.open(image_path)
        img = img.resize(target_size)  # Resize the image
        images.append(img)

# Create a figure with a grid of subplots
num_images = len(images)
rows = 2
cols = 8
fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 6))

# Flatten the axes and iterate over them
axes = axes.flatten()

# Loop through the images and plot them in the subplots
for i, (ax, img) in enumerate(zip(axes[:num_images], images)):
    ax.imshow(img)
    ax.set_xticks([])
    ax.set_yticks([])

# Hide unused subplots
for ax in axes[num_images:]:
    ax.axis('off')

# Adjust the spacing between subplots
plt.subplots_adjust(wspace=0, hspace=0)

# Display the collage
plt.show()
