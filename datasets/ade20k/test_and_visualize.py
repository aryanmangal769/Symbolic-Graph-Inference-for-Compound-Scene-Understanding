import matplotlib.pyplot as plt
import cv2
import numpy as np
import pickle as pkl
# from utils import utils_ade20k
import utils_ade20k
import pdb
from collections import Counter
import sys

# Load index with global information about ADE20K
DATASET_PATH = '/data/aryan/Seekg/Datasets/ade20k/ADE20K_2021_17_01/'
index_file = 'index_ade20k.pkl'
with open('{}/{}'.format(DATASET_PATH, index_file), 'rb') as f:
    index_ade20k = pkl.load(f)


print("File loaded, description of the attributes:")
print('--------------------------------------------')
for attribute_name, desc in index_ade20k['description'].items():
    print('* {}: {}'.format(attribute_name, desc))
print('--------------------------------------------\n')

i = 20543 #16899 # 16899, 16964
# i = index_ade20k['scene'].index('isc')
# print(i)

nfiles = len(index_ade20k['filename'])
file_name = index_ade20k['filename'][i]
num_obj = index_ade20k['objectPresence'][:, i].sum()
num_parts = index_ade20k['objectIsPart'][:, i].sum()
count_obj = index_ade20k['objectPresence'][:, i].max()
obj_id = np.where(index_ade20k['objectPresence'][:, i] == count_obj)[0][0]
obj_name = index_ade20k['objectnames'][obj_id]
full_file_name = '{}/{}'.format(index_ade20k['folder'][i], index_ade20k['filename'][i])
print("The dataset has {} images".format(nfiles))
print("The image at index {} is {}".format(i, file_name))
print("It is located at {}".format(full_file_name))
print("It happens in a {}".format(index_ade20k['scene'][i]))
print("It has {} objects, of which {} are parts".format(num_obj, num_parts))
print("The most common object is object {} ({}), which appears {} times".format(obj_name, obj_id, count_obj))

scene_counter =Counter(index_ade20k['scene'])
scenes = dict(scene_counter.most_common(30))

# Plotting histogram for the 30 most common scenes
top_scenes, counts = zip(*scenes.items())

# plt.figure(figsize=(12, 6))
# plt.bar(scenes, counts, color='skyblue')
# plt.xlabel('Scene')
# plt.ylabel('Count')
# plt.title('Top 30 Scene Distribution in ADE20K Dataset')
# plt.xticks(rotation=45, ha='right')
# plt.tight_layout()
# plt.show()
# plt.savefig('ADE_scene_histogram.png')


highway_indices = []
count = 0

for i, scene in enumerate(index_ade20k["scene"]):
    # print(scene)
    if scene == "/living_room":
        print("#")
        highway_indices.append(i)
        count += 1

        if count == 5:
            break

for scene in highway_indices:
    root_path = DATASET_PATH.replace('ADE20K_2021_17_01/', '')
    full_file_name = '{}/{}'.format(index_ade20k['folder'][i], index_ade20k['filename'][scene])
    try:
        info = utils_ade20k.loadAde20K('{}/{}'.format(root_path, full_file_name))
    except:
        continue

    print(info['img_name'])

sys.exit()



root_path = DATASET_PATH.replace('ADE20K_2021_17_01/', '')

# This function reads the image and mask files and generate instance and segmentation
# masks
info = utils_ade20k.loadAde20K('{}/{}'.format(root_path, full_file_name))


img = cv2.imread(info['img_name'])[:,:,::-1]

# Extract object information
all_objects = info['objects']['class']
all_polygons = info['objects']['polygon']

# Filter polygons based on area
min_area_threshold = 15000  # Adjust the threshold as needed
max_area  = img.shape[0] * img.shape[1]
filtered_objects = []
filtered_polygons = []

for obj, poly in zip(all_objects, all_polygons):
    area = cv2.contourArea(np.array(list(zip(poly['x'], poly['y']))))
    # print(area)
    if area > min_area_threshold and area < max_area/3:
        filtered_objects.append(obj)
        filtered_polygons.append(poly)

print("The image has {} objects, of which {} are parts".format(len(filtered_objects), len(filtered_polygons)))

# img = cv2.imread(info['img_name'])[:,:,::-1]
# Plotting
fig, ax = plt.subplots(figsize=(15, 5))
ax.imshow(img)

# Plot bounding boxes around filtered polygons
for obj, poly in zip(filtered_objects, filtered_polygons):
    x, y, w, h = min(poly['x']), min(poly['y']), max(poly['x']) - min(poly['x']), max(poly['y']) - min(poly['y'])
    rect = plt.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    ax.text(x, y, obj, color='r', verticalalignment='top', bbox={'color': 'white', 'alpha': 0.7, 'pad': 1})

ax.axis('off')
plt.savefig('test.png')
plt.show()







# img = cv2.imread(info['img_name'])[:,:,::-1]
# seg = cv2.imread(info['segm_name'])[:,:,::-1]
# seg_mask = seg.copy()

# # The 0 index in seg_mask corresponds to background (not annotated) pixels
# seg_mask[info['class_mask'] != obj_id+1] *= 0
# plt.figure(figsize=(15,5))

# plt.imshow(np.concatenate([img, seg, seg_mask], 1))
# plt.savefig('test.png') 
# plt.axis('off')
# if len(info['partclass_mask']):
#     plt.figure(figsize=(5*len(info['partclass_mask']), 5))
#     plt.title('Parts')
#     plt.imshow(np.concatenate(info['partclass_mask'],1))
#     plt.axis('off')

