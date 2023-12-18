import matplotlib.pyplot as plt
import cv2
import numpy as np
import pickle as pkl
import utils_ade20k
from collections import Counter
from tqdm import tqdm

# Load index with global information about ADE20K
DATASET_PATH = '/data/aryan/Seekg/Datasets/ade20k/ADE20K_2021_17_01/'
index_file = 'index_ade20k.pkl'
with open('{}/{}'.format(DATASET_PATH, index_file), 'rb') as f:
    index_ade20k = pkl.load(f)

nfiles = len(index_ade20k['filename'])

scene_counter = Counter(index_ade20k['scene'])
top_scenes = dict(scene_counter.most_common(30))

# Create a dictionary to store Counter for top objects in each scene
scene_objects_counter = {}

for i in tqdm(range(nfiles)):
    file_name = index_ade20k['filename'][i]
    num_obj = index_ade20k['objectPresence'][:, i].sum()
    num_parts = index_ade20k['objectIsPart'][:, i].sum()
    count_obj = index_ade20k['objectPresence'][:, i].max()
    obj_id = np.where(index_ade20k['objectPresence'][:, i] == count_obj)[0][0]
    obj_name = index_ade20k['objectnames'][obj_id]
    full_file_name = '{}/{}'.format(index_ade20k['folder'][i], index_ade20k['filename'][i])

    if index_ade20k['scene'][i] in top_scenes:
        root_path = DATASET_PATH.replace('ADE20K_2021_17_01/', '')
        try:
            info = utils_ade20k.loadAde20K('{}/{}'.format(root_path, full_file_name))
        except:
            continue
        img = cv2.imread(info['img_name'])

        # Extract object information
        all_objects = info['objects']['class']
        all_polygons = info['objects']['polygon']

        # Filter polygons based on area
        min_area_threshold = 15000  # Adjust the threshold as needed
        max_area  = img.shape[0] * img.shape[1]
        # print(img.shape)
        # print(max_area)
        filtered_objects = []
        filtered_polygons = []

        for obj, poly in zip(all_objects, all_polygons):
            area = cv2.contourArea(np.array(list(zip(poly['x'], poly['y']))))
            # print(area)
            if area > min_area_threshold and area < max_area/3:
                filtered_objects.append(obj)
                filtered_polygons.append(poly)


        if len(filtered_objects) < 2:
            continue
        # Create a Counter for the filtered objects in the current scene
        scene_name = index_ade20k['scene'][i]
        scene_name = scene_name.replace('/', '')
        scene_name = '/'+scene_name
        if scene_name == '/isc':
            continue
        if scene_name not in scene_objects_counter:
            scene_objects_counter[scene_name] = Counter(filtered_objects)
        else:
            scene_objects_counter[scene_name] += Counter(filtered_objects)

# Now scene_objects_counter contains Counters for the top objects in each scene
# You can access them like scene_objects_counter[scene_name]
# For example, to print the top 5 objects in the first scene:
# scene_name = list(top_scenes.keys())[0]
scene_name = list(scene_objects_counter.keys())[0]
top_objects_in_scene = scene_objects_counter[scene_name].most_common(5)
print(f"Top 5 objects in scene '{scene_name}': {top_objects_in_scene}")


image_info_dict = {}
for i in tqdm(range(nfiles)):
    file_name = index_ade20k['filename'][i]
    num_obj = index_ade20k['objectPresence'][:, i].sum()
    num_parts = index_ade20k['objectIsPart'][:, i].sum()
    count_obj = index_ade20k['objectPresence'][:, i].max()
    obj_id = np.where(index_ade20k['objectPresence'][:, i] == count_obj)[0][0]
    obj_name = index_ade20k['objectnames'][obj_id]
    full_file_name = '{}/{}'.format(index_ade20k['folder'][i], index_ade20k['filename'][i])

    if index_ade20k['scene'][i] in top_scenes:
        root_path = DATASET_PATH.replace('ADE20K_2021_17_01/', '')
        try:
            info = utils_ade20k.loadAde20K('{}/{}'.format(root_path, full_file_name))
        except:
            continue
        img = cv2.imread(info['img_name'])

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
        
        if len(filtered_objects) < 2:
            continue

        # Create a Counter for the filtered objects in the current scene
        scene_name = index_ade20k['scene'][i]
        scene_name = scene_name.replace('/', '')
        scene_name = '/'+scene_name
        if scene_name == '/isc':
            continue

        # Create a dictionary for image information
        image_info_dict[file_name] = {
            'scene_name': scene_name,
            'most_common_object_name': scene_objects_counter[scene_name].most_common(1)[0][0],
            'object_bboxes': [(min(poly['x']), min(poly['y']), max(poly['x']), max(poly['y']))  for poly in filtered_polygons],
            'object_names': filtered_objects
        }

image_name = list(image_info_dict.keys())[0]
print(f"Image: {image_name}")
print(f"Scene Name: {image_info_dict[image_name]['scene_name']}")
print(f"Most Common Object: {image_info_dict[image_name]['most_common_object_name']}")
print(f"Object Bounding Boxes: {image_info_dict[image_name]['object_bboxes']}")

# Save scene_objects_counter
with open('scene_objects_counter.pkl', 'wb') as f:
    pkl.dump(scene_objects_counter, f)

# Save image_info_dict
with open('image_info_dict.pkl', 'wb') as f:
    pkl.dump(image_info_dict, f)