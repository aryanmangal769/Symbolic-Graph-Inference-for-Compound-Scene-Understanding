import torch
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from PIL import Image
import os
from collections import defaultdict, Counter
from heapq import nlargest
from tqdm import tqdm
import pickle
import pdb

CLASSES = ['__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
# Load the pre-trained Faster R-CNN model
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

def get_bboxes_labels(image_path, min_objects=5):
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

    # Track unique object labels and their corresponding bounding boxes
    category_bboxes_labels = defaultdict(list)

    for box, label, score in zip(boxes, labels, scores):
        if score < 0.3:
            continue

        # Add bounding box and label to the category
        category_bboxes_labels[CLASSES[label]].append({
            'bbox': box,
            'label': CLASSES[label],
            'score': score
        })

    return category_bboxes_labels



base_dir = "/data/aryan/Seekg/Datasets/places365/Object_annotations"
with open("images_per_class.pkl", "rb") as file:
    images_per_class, top_objects_per_class = pickle.load(file)

for category in tqdm(images_per_class.keys()):
    print(category)
    print(Counter(top_objects_per_class[category] ).most_common(10))

pdb.set_trace()

# Dictionary to store bounding boxes and labels for each category

for category in tqdm(images_per_class.keys()):
    category_bboxes_labels = defaultdict(list)
    for image_path in images_per_class[category]:
        # Get bounding boxes and labels for each image
        image_bboxes_labels = get_bboxes_labels(image_path)
        category_bboxes_labels[image_path] = image_bboxes_labels

    # Save the result for the category
    output_file_name = f"{category}_bboxes_labels.pkl"
    output_file_path = os.path.join(base_dir, output_file_name)
    with open(output_file_path, "wb") as output_file:
        pickle.dump(category_bboxes_labels, output_file)