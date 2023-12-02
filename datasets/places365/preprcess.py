import torch
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from PIL import Image
import os
from collections import defaultdict, Counter
from heapq import nlargest
from tqdm import tqdm
import pickle

CLASSES = ['__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
categories = ['airfield', 'aquarium', 'army_base', 'ballroom', 'beach', 'bedroom', 'botanical_garden', 'church_outdoor', 'classroom', 'computer_room', 'food_court', 'gas_station', 'hospital_room', 'kitchen', 'parking_lot', 'restaurant', 'theater_indoor', 'waterfall']


# Load the pre-trained Faster R-CNN model
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

print("Model loaded successfully")

def detect_and_count_objects(image_path, min_objects=5):
    # Load the image
    img = Image.open(image_path).convert("RGB")
    
    # Define the transformation
    transform = T.Compose([T.ToTensor()])

    # Apply the transformation
    img_tensor = transform(img)
    img_tensor = img_tensor.unsqueeze(0)  # Add a batch dimension


    # Make prediction
    with torch.no_grad():
        prediction = model(img_tensor)

    # Get bounding boxes, labels, and scores
    boxes = prediction[0]['boxes'].cpu().numpy()
    labels = prediction[0]['labels'].cpu().numpy()
    scores = prediction[0]['scores'].cpu().numpy()

    # Track unique object labels
    unique_labels = set()

    for box, label, score in zip(boxes, labels, scores):
        if score < 0.3:
            continue
        x, y, w, h = box

        # Add the label to the set
        # print(CLASSES[label])
        unique_labels.add(CLASSES[label])
    return unique_labels

def process_dataset(dataset_dir, min_objects=3):
    class_counts = defaultdict(int)
    constraint_met_counts = defaultdict(int)
    top_objects_per_class = defaultdict(list)
    images_per_class = defaultdict(list)

    for letter in tqdm(os.listdir(dataset_dir)):
        letter_dir = os.path.join(dataset_dir, letter)
        for category in tqdm(os.listdir(letter_dir)):
            if category not in categories:
                continue
            print(category)
            category_dir = os.path.join(letter_dir, category)
            for image in tqdm(os.listdir(category_dir)):
                image_path = os.path.join(category_dir, image)

                # Count total images per class
                class_counts[category] += 1

                # Check the number of unique objects
                unique_objects = detect_and_count_objects(image_path, min_objects)
                if len(unique_objects) > min_objects:
                    # print("####")
                    constraint_met_counts[category] += 1
                    images_per_class[category].append(image_path)
                    # Count occurrences of objects per class
                    top_objects_per_class[category].extend(list(unique_objects))
    
    pickle.dump((images_per_class, top_objects_per_class), open("images_per_class_2.pkl", "wb"))
    # Display results
    print("Class name : Total images : Images following the 5-object constraint : Top objects per class")
    for category in class_counts:
        print(Counter(top_objects_per_class[category]))
        print(f"{category} : {class_counts[category]} : {constraint_met_counts[category]} : {Counter(top_objects_per_class[category]).most_common(10)}")

# Example usage
if __name__ == "__main__":
    base_dir = "/data/aryan/Seekg/Datasets/places365"
    dataset_dir = os.path.join(base_dir, "data_256")

    process_dataset(dataset_dir, min_objects=3)
