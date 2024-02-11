import matplotlib.pyplot as plt
import cv2
import numpy as np
import pickle as pkl
import utils_ade20k
from collections import Counter
from tqdm import tqdm
import os

# Load index with global information about ADE20K
DATASET_PATH = '/data/aryan/Seekg/Datasets/ade20k/ADE20K_2021_17_01/'
index_file = 'index_ade20k.pkl'
with open('{}/{}'.format(DATASET_PATH, index_file), 'rb') as f:
    index_ade20k = pkl.load(f)

nfiles = len(index_ade20k['filename'])

scene_counter = Counter(index_ade20k['scene'])
top_scenes = dict(scene_counter.most_common(30))

# Scenes to be removed
scenes_to_remove = ['/bedroom','/home_office', '/attic', '/waiting_room', '/mountain_snowy', '/hotel_room', '/building_facade', 'isc', 'bedroom', 'living_room', 'bathroom']

# Remove specified scenes from the dictionary
for scene_name in scenes_to_remove:
    top_scenes.pop(scene_name, None)
    
print(top_scenes)

# Directory to save annotated images
SAVE_DIR = 'annotated_images'
os.makedirs(SAVE_DIR, exist_ok=True)

# Load or initialize image_object_relations dictionary and last processed index
resume_file = "annotated_images/resume_info.pkl"
if os.path.exists(resume_file):
    with open(resume_file, "rb") as f:
        resume_info = pkl.load(f)
    image_object_relations = resume_info['image_object_relations']
    last_processed_index = resume_info['last_processed_index']
    processed_images_per_scene = resume_info['processed_images_per_scene']
else:
    image_object_relations = {}
    last_processed_index = 0
    processed_images_per_scene = {scene: 0 for scene in top_scenes}


# Function to annotate images
def annotate_image(img, objects, polygons):
    # Draw bounding boxes
    # for obj, poly in zip(objects, polygons):
    #     pts = np.array(list(zip(poly['x'], poly['y'])), np.int32)
    #     pts = pts.reshape((-1, 1, 2))
    #     cv2.polylines(img, [pts], isClosed=True, color=(255, 0, 0), thickness=2)
    #     cv2.putText(img, obj, (pts[0][0][0], pts[0][0][1]), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
    # return img

        # Copy the original image to avoid modifying the original
    img_with_annotations = img.copy()
    
    # Pad the image by 200 pixels from all sides
    img_with_annotations = cv2.copyMakeBorder(img_with_annotations, 200, 200, 200, 200, cv2.BORDER_CONSTANT)
    
    # Adjust polygon coordinates based on padding
    for poly in polygons:
        for point in poly['x']:
            point += 200
        for point in poly['y']:
            point += 200
    
    # Draw bounding boxes and labels
    for obj, poly in zip(objects, polygons):
        pts = np.array(list(zip(poly['x'], poly['y'])), np.int32)
        pts = pts.reshape((-1, 1, 2))
        pts += 200  # Adjust all points of the polyline

        cv2.polylines(img_with_annotations, [pts], isClosed=True, color=(255, 0, 0), thickness=2)
        cv2.putText(img_with_annotations, obj, (pts[0][0][0], pts[0][0][1]), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
    
    return img_with_annotations

# Initialize counters to keep track of processed images per scene
# processed_images_per_scene = {scene: 0 for scene in top_scenes}

spatial_relations = ["on", "next to", "behind", "in front of",  "above", "across", "below", "inside", "under", "left" , "right", "in" , "None"]

for i in tqdm(range(last_processed_index, nfiles)):
    file_name = index_ade20k['filename'][i]
    num_obj = index_ade20k['objectPresence'][:, i].sum()
    num_parts = index_ade20k['objectIsPart'][:, i].sum()
    count_obj = index_ade20k['objectPresence'][:, i].max()
    obj_id = np.where(index_ade20k['objectPresence'][:, i] == count_obj)[0][0]
    obj_name = index_ade20k['objectnames'][obj_id]
    full_file_name = '{}/{}'.format(index_ade20k['folder'][i], index_ade20k['filename'][i])

    if index_ade20k['scene'][i] in top_scenes:
        print(processed_images_per_scene[index_ade20k['scene'][i]])
        if processed_images_per_scene[index_ade20k['scene'][i]] >= 5:
            print(processed_images_per_scene[index_ade20k['scene'][i]])
            continue
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
        
        if len(filtered_objects) < 3:
            continue

        # Create a Counter for the filtered objects in the current scene
        scene_name = index_ade20k['scene'][i]
        scene_name = scene_name.replace('/', '')
        scene_name = '/'+scene_name
        if scene_name == '/isc':
            continue

        # remove ambiguous scenes 
        if scene_name == '/home_office':
            scene_name = '/office'
        if scene_name == '/attic':
            scene_name = '/bedroom'
        if scene_name == '/waiting_room':
            scene_name = '/living_room'
        if scene_name == '/mountain_snowy':
            scene_name = '/mountain'
        if scene_name == '/hotel_room':
            scene_name = '/bedroom'
        if scene_name == '/building_facade':
            scene_name = '/skyscraper'
        
        # if scene_name == '/highway':
        #     continue

        # Annotate image

        print(scene_name)

        element_counts = {}

        converted_list = []
        for element in filtered_objects:
            if element not in element_counts:
                element_counts[element] = 0
            converted_list.append(f"{element}_{element_counts[element]}")
            element_counts[element] += 1
        
        filtered_objects = converted_list

        annotated_img = annotate_image(img.copy(), filtered_objects, filtered_polygons)

        # Save annotated image
        save_path = os.path.join(SAVE_DIR, "test.jpg")
        cv2.imwrite(save_path, annotated_img)

        # Increment the counter for processed images for the current scene
        processed_images_per_scene[scene_name] += 1

        # Prompt user to select objects and relations
        print(f"Image saved: {save_path}")
        # Add code for user interaction here (e.g., input for object selection, relation, etc.)

        # Initialize lists to store selected objects and relations
        selected_relations = []

        # Loop to allow user to select multiple relations
        while True:
            try:
                selected_objects = []
                print("\nSelect objects:")
                for j, obj in enumerate(filtered_objects):
                    print(f"{j + 1}. {obj}")

                print("\nSelect relation:")
                for j, relation in enumerate(spatial_relations):
                    print(f"{j + 1}. {relation}")

                selected_obj_indices = input("Enter indices of (objects, object,  relation) separated by commas (e.g., 1,2, 3) or press 'q' to quit: ")
                
                # Check if user wants to quit
                if selected_obj_indices.lower() == 'q':
                    break
                
                selected_indices = [int(idx) - 1 for idx in selected_obj_indices.split(",")]

                selected_obj_indices = selected_indices[:-1]

                for idx in selected_obj_indices:
                    selected_objects.append(filtered_objects[idx])

                selected_relation_idx = selected_indices[-1]
                selected_relation = spatial_relations[selected_relation_idx]
                
                print(f"Selected objects: {selected_objects}")
                print(f"Selected relation: {selected_relation}")
                
                if input("'n' to reselect: ").lower() == 'n':
                    continue
                

                # Append selected objects and relation to the lists
                selected_relations.append({"objects": selected_objects, "relation": selected_relation})
            except("KeyboardInterrupt"):
                print("Invalid input! Try again.")
                continue

        # Store the selected objects and relations in the dictionary
        image_object_relations[file_name] = selected_relations

        # After processing all images, you can save the dictionary and last processed index to a file
        with open("annotated_images/image_object_relations.pkl", "wb") as f:
            pkl.dump(image_object_relations, f)

        # Update last processed index
        last_processed_index = i + 1
        with open(resume_file, "wb") as f:
            pkl.dump({'image_object_relations': image_object_relations, 'last_processed_index': last_processed_index, 'processed_images_per_scene': processed_images_per_scene}, f)
