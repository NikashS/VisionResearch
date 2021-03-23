import numpy as np
import os
import pandas as pd
import pickle
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPRegressor 
from sklearn.feature_extraction.text import TfidfVectorizer
import sys
import torch
from tqdm import tqdm

sys.path.append(os.getcwd() + '/..')
sys.path.append(os.getcwd() + '/../..')
from CLIP import clip
from train_test_split import train_filenames, test_filenames

print ('starting...')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model, preprocess = clip.load('ViT-B/32')

seen_labels = set([x+1 for x in range(160)])
unseen_labels = set([x+161 for x in range(40)])

def get_image_features(dataset_path, train=True):
    all_features = []
    all_labels = []
    with torch.no_grad():
        for bird_folder in tqdm(os.listdir(dataset_path)):
            bird_id = int(bird_folder.split('.')[0])
            if bird_id in seen_labels:
                all_labels += [bird_id]

                images = []
                subdirectory = f'{dataset_path}{bird_folder}/'
                for image_filename in os.listdir(subdirectory):
                    if (train and image_filename in train_filenames) or (not train and image_filename in test_filenames):
                        image_path = f'{subdirectory}{image_filename}'
                        image = Image.open(image_path).convert('RGB')
                        image_input = preprocess(image).unsqueeze(0).to(device)
                        images += [image_input]

                images = torch.cat(images, 0)
                features = model.encode_image(images)
                features = np.resize(features.cpu().numpy().flatten(), 15360)
                all_features += [features]
    return all_features, all_labels

def get_text_features(dataset_path, seen=True):
    all_features = []
    all_labels = []

    corpus = []
    for bird_filename in os.listdir(dataset_path):
        bird_id = int(bird_filename.split('.')[0])
        if (seen and bird_id in seen_labels) or (not seen and bird_id in unseen_labels):
            all_labels += [bird_id]

            bird_path = f'{dataset_path}{bird_filename}'
            bird_file = open(bird_path, 'r', errors='ignore')
            bird_text = bird_file.read()
            corpus += [bird_text]

    vectorizer = TfidfVectorizer(use_idf=True, max_features=1000)
    vectors = vectorizer.fit_transform(corpus)
    all_features = vectors.todense()

    return all_features, all_labels

def main_pickled():
    train_features, train_labels=pickle.load(open('pickle/train_data.pkl', 'rb'))
    test_features, test_labels = pickle.load(open('pickle/test_data.pkl', 'rb'))
    classifier = pickle.load(open('pickle/logres_image_classifier.pkl', 'rb'))
    return (train_features, train_labels, test_features, test_labels, classifier)

# images_directory = r'/localtmp/data/cub/CUB_200_2011/images/'
# train_features, train_labels = get_image_features(images_directory, train=True)
# pickle.dump((train_features, train_labels), open('pickle/train_data.pkl', 'wb'))
# test_features, test_labels = get_image_features(directory, train=False)
# pickle.dump((test_features, test_labels), open('pickle/test_data.pkl', 'wb'))

# classifier = LogisticRegression(C=0.316, max_iter=100, solver='sag', n_jobs=3000)
# classifier.fit(train_features, train_labels)
# pickle.dump(classifier, open('pickle/logres_image_classifier.pkl', 'wb'))
# predictions = classifier.predict(test_features)
# accuracy = np.mean((test_labels == predictions).astype(np.float)) * 100.
# print (f'Accuracy: {accuracy:.3f}')

train_features, train_labels, test_features, test_labels, classifier = main_pickled()

wikipedia_directory = r'/localtmp/data/cub/birds_wikipedia/'
seen_wikipedia_features, seen_wikipedia_labels = get_text_features(wikipedia_directory, seen=True)
seen_label_indices = np.asarray(seen_wikipedia_labels) - 1
ordered_seen_wikipedia_features = seen_wikipedia_features[seen_label_indices]
perceptron = MLPRegressor(max_iter=100, verbose=1, learning_rate='adaptive')
perceptron.fit(ordered_seen_wikipedia_features, classifier.coef_)





