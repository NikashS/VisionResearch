import numpy as np
import os
import pandas as pd
import pickle
from PIL import Image
from skimage.measure import block_reduce
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, auc
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neural_network import MLPRegressor 
from sklearn.preprocessing import normalize, LabelBinarizer, StandardScaler
import sys
import torch
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(os.getcwd() + '/..')
sys.path.append(os.getcwd() + '/../..')
from CLIP import clip
from train_test_split import train_filenames, test_filenames

print ('starting...')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model, preprocess = clip.load('ViT-B/32')

seen_labels = set([x+1 for x in range(160)])
unseen_labels = set([x+161 for x in range(40)])

def get_image_features(dataset_path, train=True, seen=True):
    all_features = []
    all_labels = []
    with torch.no_grad():
        for bird_folder in tqdm(os.listdir(dataset_path)):
            bird_id = int(bird_folder.split('.')[0])
            if (seen and bird_id in seen_labels) or (not seen and bird_id in unseen_labels):
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
                features = features.cpu().numpy()
                # features = np.resize(features.flatten(), 15360)
                features = block_reduce(features, (len(features), 1), np.mean).flatten()
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
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    vectors = vectorizer.fit_transform(corpus)
    # df = pd.DataFrame(vectors[0].T.todense(), index=vectorizer.get_feature_names(), columns=["tfidf"])
    # df = df.sort_values(by=["tfidf"],ascending=False)
    all_features = vectors.todense()
    return all_features, all_labels

def generate_noise_train_data(wikipedia_features, image_embeddings, multiplier=0):
    min_weight = np.absolute(image_embeddings).min()*1000
    for i in range(multiplier):
        noise = np.random.normal(0, min_weight, image_embeddings.shape)
        image_embeddings_noise = image_embeddings + noise
        image_embeddings = np.concatenate((image_embeddings, image_embeddings_noise))
        wikipedia_features = np.concatenate((wikipedia_features, wikipedia_features))
    return wikipedia_features, image_embeddings

def accuracy(classifier, features, labels):
    predictions = classifier.predict(features)
    lb = LabelBinarizer()
    lb.fit(labels)
    labels_binarized = lb.transform(labels)
    predictions_binarized = lb.transform(predictions)
    print (f'ROC AUC Score: {roc_auc_score(labels_binarized, predictions_binarized, average="weighted")}')
    print (f'PR AUC Score: {average_precision_score(labels_binarized, predictions_binarized)}')
    # print (predictions)
    accuracy = np.mean((labels == predictions).astype(np.float)) * 100.
    return accuracy

# images_directory = r'/localtmp/data/cub/CUB_200_2011/images/'
# train_features, train_labels = get_image_features(images_directory, train=True)
# pickle.dump((train_features, train_labels), open('pickle_test/train_data.pkl', 'wb'))
# test_features, test_labels = get_image_features(images_directory, train=False)
# pickle.dump((test_features, test_labels), open('pickle_test/test_data.pkl', 'wb'))
# unseen_features, unseen_labels = get_image_features(images_directory, train=False, seen=False)
# pickle.dump((unseen_features, unseen_labels), open('pickle_test/test_unseen_data.pkl', 'wb'))
train_features, train_labels = pickle.load(open('pickle/train_data.pkl', 'rb'))
test_features, test_labels = pickle.load(open('pickle/test_data.pkl', 'rb'))
unseen_features, unseen_labels = pickle.load(open('pickle/test_unseen_data.pkl', 'rb'))

all_features = np.concatenate((train_features, unseen_features))
all_labels = np.concatenate((train_labels, unseen_labels))

# classifier = LogisticRegression(C=0.316, max_iter=500, solver='sag', n_jobs=1000, verbose=1)
# classifier.fit(all_features, all_labels)
# pickle.dump(classifier, open('pickle/logres_image_classifier_ground_truth.pkl', 'wb'))
classifier = pickle.load(open('pickle/logres_image_classifier_ground_truth.pkl', 'rb'))
all_real_weights = np.concatenate((classifier.coef_, np.asarray([classifier.intercept_]).T / 1e5), axis=1)

# classifier.fit(train_features, train_labels)
# pickle.dump(classifier, open('pickle_test/logres_image_classifier.pkl', 'wb'))
classifier = pickle.load(open('pickle/logres_image_classifier.pkl', 'rb'))

seen_classifier_intercepts = np.asarray([classifier.intercept_]).T / 1e5
seen_image_embeddings = np.concatenate((classifier.coef_, seen_classifier_intercepts), axis=1)

wikipedia_directory = r'/localtmp/data/cub/birds_wikipedia/'
seen_wikipedia_features, seen_wikipedia_labels = get_text_features(wikipedia_directory, seen=True)
unseen_wikipedia_features, unseen_wikipedia_labels = get_text_features(wikipedia_directory, seen=False)
seen_label_indices = np.asarray(seen_wikipedia_labels) - 1
unseen_label_indices = np.asarray(unseen_wikipedia_labels) - 161
ordered_seen_wikipedia_features = seen_wikipedia_features[seen_label_indices]
ordered_unseen_wikipedia_features = unseen_wikipedia_features[unseen_label_indices]

ordered_seen_wikipedia_features = normalize(ordered_seen_wikipedia_features, norm="l2")
ordered_unseen_wikipedia_features = normalize(ordered_unseen_wikipedia_features, norm="l2")

more_features, more_embeddings = generate_noise_train_data(ordered_seen_wikipedia_features, seen_image_embeddings)
# more_features, more_embeddings = (ordered_seen_wikipedia_features, seen_image_embeddings)

seen_loss = []
unseen_loss = []
for i in range(250, 10000, 250):
    perceptron = MLPRegressor(
        max_iter=i,
        alpha=0.001,
        learning_rate='adaptive',
        tol=-1*float('inf'),
        verbose=0,
    )
    perceptron.fit(more_features, more_embeddings)
    all_wikipedia_features = np.concatenate((ordered_seen_wikipedia_features, ordered_unseen_wikipedia_features), axis=0)
    all_image_embeddings = perceptron.predict(all_wikipedia_features)
    all_predicted_weights = all_image_embeddings

    seen_loss += [np.mean(euclidean_distances(all_real_weights[:160], all_predicted_weights[:160]))]
    unseen_loss += [np.mean(euclidean_distances(all_real_weights[160:200], all_predicted_weights[160:200]))]

plt.plot(list(range(250, 10000, 250)), seen_loss, label="Seen Losses")
plt.plot(list(range(250, 10000, 250)), unseen_loss, label="Unseen Losses")
plt.legend()
plt.savefig('loss_curves_test.png')