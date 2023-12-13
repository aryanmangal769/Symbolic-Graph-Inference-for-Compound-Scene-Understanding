import os
import pickle
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pdb

# Load the dictionary containing bounding boxes and labels
base_dir = "/data/aryan/Seekg/Datasets/places365/Object_annotations"
category = "aquarium"  # Specify the category you want to visualize
output_file_name = f"{category}_bboxes_labels.pkl"
output_file_path = os.path.join(base_dir, output_file_name)

with open(output_file_path, "rb") as file:
    category_bboxes_labels = pickle.load(file)

# Choose an image from the category
iterator = iter(category_bboxes_labels.keys())

# print(len(category_bboxes_labels))
image_path = next(iterator)
image_path = next(iterator)

print(image_path)
# Load the image
img = Image.open(image_path).convert("RGB")

# Get bounding boxes and labels for the image
image_bboxes_labels = category_bboxes_labels[image_path]
# print(image_bboxes_labels)
# Plot the image with bounding boxes
fig, ax = plt.subplots(1)
ax.imshow(img)

for obj in image_bboxes_labels:
    print(obj)
    for obj_data in image_bboxes_labels[obj]:
        bbox = obj_data['bbox']
        label = obj_data['label']
        score = obj_data['score']
        # print(bbox)

        # Add bounding box to the image
        rect = patches.Rectangle(
            (bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1],
            linewidth=1, edgecolor='r', facecolor='none', label=f'{label}: {score:.2f}'
        )
        ax.add_patch(rect)

        # Add label text to the image
        ax.text(bbox[0], bbox[1] - 5, f'{label}: {score:.2f}', color='r', fontsize=8, weight='bold')

# Display the image with bounding boxes and labels
plt.axis('off')
# plt.show()
plt.savefig('q4_.png')
