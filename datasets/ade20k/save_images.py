import os
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
top_scenes = np.unique([scene.replace('/','') for scene in top_scenes.keys()])

# Create a dictionary to store Counter for top objects in each scene
scene_objects_counter = {}

# Define the output folder where you want to save the images
output_folder = 'results/ChatGPT_imgs'

# Create a dictionary to keep track of the number of images saved for each scene
scene_image_count = {}

# Loop through the dataset
for i in tqdm(range(nfiles)):
    file_name = index_ade20k['filename'][i]
    scene = index_ade20k['scene'][i]
    scene = scene.replace('/','')
    # Check if the scene is in the top scenes
    if scene.replace('/','') in top_scenes:
        # Create a folder for the scene if it doesn't exist
        scene_folder = os.path.join(output_folder, scene)
        os.makedirs(scene_folder, exist_ok=True)

        # Check if the number of images saved for the scene is less than 5
        if scene_image_count.get(scene, 0) < 5:
            # Update the count for the scene
            scene_image_count[scene] = scene_image_count.get(scene, 0) + 1
            full_file_name = '{}/{}'.format(index_ade20k['folder'][i], index_ade20k['filename'][i])
            # Load and save the image
            root_path = DATASET_PATH.replace('ADE20K_2021_17_01/', '')
            info = utils_ade20k.loadAde20K('{}/{}'.format(root_path, full_file_name))
            img = cv2.imread(info['img_name'])
            img_save_path = os.path.join(scene_folder, f'{scene_image_count[scene]}_{file_name}')
            cv2.imwrite(img_save_path, img)
