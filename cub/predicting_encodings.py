import numpy as np
import os
import pandas as pd
import pickle
from PIL import Image
from sklearn.linear_model import LogisticRegression
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

def get_image_features(dataset_path, train=True):
    all_features = []
    all_labels = []
    with torch.no_grad():
        for bird_folder in tqdm(os.listdir(dataset_path)):
            bird_id = int(bird_folder.split('.')[0])
            all_labels.append(bird_id)

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

directory = r'/localtmp/data/cub/CUB_200_2011/images/'

# train_features, train_labels = get_image_features(directory, train=True)
# pickle.dump((train_features, train_labels), open('pickle/train_data.pkl', 'wb'))
train_features, train_labels=pickle.load(open('pickle/train_data.pkl', 'rb'))

# test_features, test_labels = get_image_features(directory, train=False)
# pickle.dump((test_features, test_labels), open('pickle/test_data.pkl', 'wb'))
test_features, test_labels = pickle.load(open('pickle/test_data.pkl', 'rb'))

print ('fitting...')
classifier = LogisticRegression(random_state=0, C=0.5, max_iter=100, verbose=0, n_jobs=3000, solver='sag')
classifier.fit(train_features, train_labels)
pickle.dump(classifier, open('pickle/logres_image_classifier.pkl', 'wb'))

predictions = classifier.predict(test_features)
accuracy = np.mean((test_labels == predictions).astype(np.float)) * 100.
print (f'Accuracy: {accuracy:.3f}')






