import pickle
from collections import Counter
import pdb
from sklearn.metrics import top_k_accuracy_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder

def get_accuracy(targets,predictions ):
    accuracy = 0
    for i in range(len(predictions)):
        
        if targets[i] in predictions[i]:
            accuracy += 1

    return accuracy/len(predictions)

file_path = "results/chat_gpt_test.pkl"

# Load the data from the pickle file
with open(file_path, 'rb') as f:
    loaded_data = pickle.load(f)

# Unpack the loaded data
active_idxs, actual_verbs = loaded_data

# base_dir = "/data/aryan/Seekg/Datasets/places365/Object_annotations"
with open("images_per_class.pkl", "rb") as file:
    images_per_class, top_objects_per_class = pickle.load(file)
    
# actual_verbs, active_idxs = actual_verbs[:100], active_idxs[:100]


# Function to classify image based on objects
def classify_image(objects, top_objects_per_class):
    # Find the category with the most common objects in the image
    category_counts = Counter()
    objects = [obj.split('_')[0] for obj in objects]
    for category, top_objects in top_objects_per_class.items():
        top_objects = Counter(top_objects).most_common(50)
        top_objects = [obj[0] for obj in top_objects]
        common_objects = set(objects) & set(top_objects)
        category_counts[category] = len(common_objects)

    # print(category_counts)
    # Get the category with the highest count
    classified_category = [obj[0] for obj in category_counts.most_common(5)]
    return classified_category

# Classify each image in active_idxs
classified_categories = []
for objects_per_image in active_idxs:
    # Classify the image based on its objects
    category = classify_image(objects_per_image, top_objects_per_class)
    classified_categories.append(category)

# print(classified_categories)

print("F1 Score: ", f1_score(actual_verbs, [c[0] for c in classified_categories], average='weighted'))
# Print or use the classified categories as needed
print(get_accuracy(actual_verbs, classified_categories))