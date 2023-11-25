import numpy as np
import pickle
import os
from PIL import Image
from scipy.stats import multivariate_normal
import time
import cv2
from collections import Counter
import sys
sys.path.append('.')
# from utils.valid_interactions import colormap_interactions
#Create a dataset class to load an image and its corresponding pickle file
from datasets.epic_kitchens import EPIC_Kitchens


class Ego_Metric_training_dataset():
    def __init__(self, Ego_Metric_dataset_path):
        self.main_dir = Ego_Metric_dataset_path
        # self.samples_txt = os.path.join(self.main_dir, 'samples.txt')
        self.img_dir = 'selected_plus_guided_rgb'
        self.label_2d = '50_classes_2d_output_labels'
        self.label_3d = 'aff_on_3d'
        self.valid_verbs = ['take', 'remove', 'put', 'insert', 'throw', 'wash', 'dry', 'open', 'turn-on', 
                            'close', 'turn-off', 'mix', 'fill', 'add', 'cut', 'peel', 'empty', 
                            'shake', 'squeeze', 'press', 'cook', 'move', 'adjust', 'eat', 
                            'drink', 'apply', 'sprinkle', 'fold', 'sort', 'clean', 'slice', 'pick']
        self.height = 480
        self.width = 854
        self.size = 500
        self.samples = self.obtain_samples()
        print('Number of samples: ', len(self.samples))
        # self.pos = self.get_pos_for_gaussian()
        # self.gaussian = self.get_gaussian()
        # self.colormap_interactions = colormap_interactions()

    def obtain_samples(self):
        samples = []
        label_main_dir = os.path.join(self.main_dir, 'EPIC_Aff_50_classes_2d_output_labels')
        for kitchen in os.listdir(label_main_dir):
            if kitchen != 'samples.txt':
                if not os.path.exists(os.path.join(label_main_dir, kitchen, self.label_2d)):
                    print("No label_2d for kitchen: ", kitchen)
                    continue
                for sample in os.listdir(os.path.join(label_main_dir, kitchen, self.label_2d)):
                    sample_id = sample.split('.')[0]
                    samples.append(kitchen + '/' + sample_id)
        return samples

    def __len__(self):
        return len(self.samples)  

    def __getitem__(self, idx):
        kitchen, sample_id = self.samples[idx].split('/')
        #Load the image
        img_main_dir = os.path.join(self.main_dir, "EPIC_Aff_images")
        label_main_dir = os.path.join(self.main_dir, 'EPIC_Aff_50_classes_2d_output_labels')
        img_path = os.path.join(img_main_dir, kitchen, self.img_dir, sample_id + '.jpg')
        # img = cv2.imread(img_path)
        #Load the labels
        label_2d_path = os.path.join(label_main_dir, kitchen, self.label_2d, sample_id + '.pkl')
        with open(label_2d_path, 'rb') as f:
            data_2d = pickle.load(f)
            verbs_data = data_2d['verbs']
            noun_data = data_2d['nouns']

        return verbs_data, noun_data ,img_path
    
    

def verbs_per_noun(verbs, nouns):
    hist = {}
    for i in range(len(verbs)):
        verb = verbs[i]
        noun = nouns[i]
        if noun not in hist.keys():
            hist[noun] = {}
        if verb not in hist[noun]:
            hist[noun][verb] = 1
        else:
            hist[noun][verb] += 1
    return hist

def get_cooccurance(nouns_unique):
    relations = {}

    for noun_list in nouns_unique:
        for i in range(len(noun_list)):
            noun = noun_list[i]
            if noun not in relations:
                relations[noun] = {}

            for j in range(len(noun_list)):
                if i != j:
                    co_occurred_noun = noun_list[j]
                    if co_occurred_noun not in relations[noun]:
                        relations[noun][co_occurred_noun] = 1
                    else:
                        relations[noun][co_occurred_noun] += 1

    return relations


def count_nouns(nouns):
    noun_counts = {}
    
    for noun in nouns:
        if noun in noun_counts:
            noun_counts[noun] += 1
        else:
            noun_counts[noun] = 1
    
    for noun, count in noun_counts.items():
        print(f"{noun}: {count} times")

def get_verb_for_noun(verbs, nouns, noun):
    verb_counts = Counter()
    
    for i in range(len(nouns)):
        if nouns[i] == noun:
            verb_counts[verbs[i]] += 1
    
    most_common_verbs = verb_counts.most_common(2)
    if len(most_common_verbs) >= 2:
        most_common_verb, count_most_common = most_common_verbs[0]
        # print(most_common_verbs)
        second_most_common_verb, count_second_most_common = most_common_verbs[1]        
        # if count_most_common >= 1.2 * count_second_most_common :
        if count_most_common > 10:
        # if count_most_common >  count_second_most_common :
            return most_common_verb
        else:
            return None
    else:
        return None

if __name__ == "__main__":
    base_dir = '/data/aryan/Seekg/Datasets/epic_kitchens_affordances/data'
    data = Ego_Metric_training_dataset(base_dir)
    epic_kitchens = EPIC_Kitchens(base_dir, None)


    # objects = ['plate', 'cup', 'pan', 'knife', 'meat', 'board:chopping']
    # verbs_per_object = [['wash','take'],['wash','take'],['wash','take'],['wash','take'],['cut','take'],['wash','take']]

    # objects = ['board:chopping']
    # verbs_per_object = [['wash','move','scrape']]

    # objects = ['meat','cup','rice','pan','board:chopping','chicken']
    # verbs_per_object = [['cut','mix'],['wash','shake'],['pour','mix'],['wash','insert'],['move','wash'],['break','wash']]0

    objects = ['meat','cup','rice','pan']
    verbs_per_object = [['cut','mix'],['wash','shake'],['pour','mix'],['wash','insert']]

    dataset = {}
    verbs = []
    nouns = []
    nouns_unique = []

    for i in range(len(data)):
        # verb, noun ,path = data[i]
        # print("Working on sample: ", i)

        for j, object in enumerate(objects):
            verb, noun ,path = data[i]

            if object in noun:
                verb_for_noun = get_verb_for_noun(verb, noun, object)
                if verb_for_noun in verbs_per_object[j]:
                    if object not in dataset:
                        dataset[object] = {}

                    if verb_for_noun not in dataset[object]:
                        dataset[object][verb_for_noun] = []
                    # print(path)
                    sample_id = path.split('/')[-1].split('.')[0]
                    subset = sample_id.split('_')[0] + '_' + sample_id.split('_')[1]
                    # print(subset)
                    if epic_kitchens.has_sparse_annotations(path.split('/')[-1], subset):
                        sample = path.split('/')[-3] + '/' + path.split('/')[-1].split('.')[0]
                        dataset[object][verb_for_noun].append(sample)
        

        # verb, noun ,path = data[i]  
        # verbs.extend(verb)
        # nouns.extend(noun)

        # nouns_unique.append(np.unique(noun))
        # break


    # Save the 'dataset' dictionary to the specified file
    with open(os.path.join(base_dir,'scene_based_affordance.pkl'), 'wb') as file:
        pickle.dump(dataset, file)

    # Print the dataset subset along with the nu,mber of images for each verb
    for object, verb_dict in dataset.items():
        print(f"Object: {object}")
        for verb, img_paths in verb_dict.items():
            print(f"  Verb: {verb}")
            # print(img_paths)
            print(f"  Number of Images: {len(img_paths)}")

    # print(np.unique(verbs))
    # print(len(np.unique(nouns)))

    ## Countes the number of time a noun appears in the dataset
    # count_nouns(nouns)

    ## Count the number of verbs per noun
    # hist = verbs_per_noun(verbs, nouns)
    # print(hist)

    ## Get the cooccured with for every noun
    # noun_relations = get_cooccurance(nouns_unique)
    # print(noun_relations)
    
    ## Print the cooccured nouns for the specified objects
    # for noun, relations in noun_relations.items():
    #     if noun in objects:
    #         print(f"Noun: {noun}")
    #         print(f"  Relations: {relations}")
    
    ## Visualisation of the verb regions for the given image. (The visualization function are removed from thsi file they are present in data.py)
    # img, masks = data[16]
    # data.visualize(img, masks, 'cut')
