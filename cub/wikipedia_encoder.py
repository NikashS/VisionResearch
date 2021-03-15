import numpy as np
import os
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import sys
import torch

print ('starting...')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_features(dataset_path):
    all_features = []
    all_labels = []

    documents = []

    for bird in os.listdir(dataset_path):
        bird_id = int(bird.split('.')[0])
        all_labels.append(bird_id)

        filename = dataset_path + bird
        f = open(filename, 'r', errors='ignore')
        text = f.read()
        documents.append(text)

    vectorizer = TfidfVectorizer(use_idf=True, max_features=1000)
    vectors = vectorizer.fit_transform(documents)
    all_features = vectors.todense()

    # df = pd.DataFrame(vectors[0].T.todense(), index=vectorizer.get_feature_names(), columns=["tfidf"]) 
    # df = df.sort_values(by=["tfidf"],ascending=False)
    # print (df)

    return all_features, all_labels

train_directory = r'/localtmp/data/cub/birds_wikipedia/'

train_features, train_labels = get_features(train_directory)

classifier = LogisticRegression(random_state=0, C=0.316, max_iter=1000, verbose=0)
classifier.fit(train_features, train_labels)

filename = 'pickle/wikipedia_encoder.pkl'
pickle.dump(classifier, open(filename, 'wb'))
