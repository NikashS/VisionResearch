import numpy as np
import os
from PIL import Image
from sklearn.linear_model import LogisticRegression
import sys
import torch
from tqdm import tqdm

sys.path.append(os.getcwd() + '/..')
from CLIP import clip
from prompt_templates import prompt_templates_openai, subset_prompt_templates_openai
from wnid_dictionaries import wnid_to_labels_openai, wnid_to_short_hyponyms

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model, preprocess = clip.load('ViT-B/32')

def get_features(dataset_path):
    all_features = []
    all_labels = []

    with torch.no_grad():
        for wnid in os.listdir(dataset_path):
            subdirectory = dataset_path + wnid + r'/'
            images = []

            # Preprocess images
            for filename in os.listdir(subdirectory):
                filepath = subdirectory + filename
                image = Image.open(filepath).convert('RGB')
                image_input = preprocess(image).unsqueeze(0).to(device)
                images += [image_input]
            images = torch.cat(images, 0)

            image_features = model.encode_image(images)
            label = wnid_to_labels_openai[wnid]
            all_features.append(features)
            all_labels.append(label)

    return torch.cat(all_features).cpu().numpy(), torch.cat(all_labels).cpu().numpy()

train_directory = r'/localtmp/data/imagenet256/train/'
test_directory = r'/localtmp/data/imagenet256/val/'

train_features, train_labels = get_features(train_directory)
test_features, test_labels = get_features(test_directory)

classifier = LogisticRegression(random_state=0, C=0.316, max_iter=1000, verbose=1)
classifier.fit(train_features, train_labels)

predictions = classifier.predict(test_features)
accuracy = np.mean((test_labels == predictions).astype(np.float)) * 100.
print (f'Accuracy: {accuracy:.3f}')

