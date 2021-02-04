import numpy as np
import os
import pickle
from PIL import Image
import sys
import torch
from tqdm import tqdm

sys.path.append(os.getcwd() + '/..')
from CLIP import clip
from prompt_templates import imagenet_templates_openai
from wnid_dictionaries import wnid_to_labels_openai

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32")
resnet_model, _ = clip.load("RN50")

# Model parameters of ViT and RN are the same
input_resolution = clip_model.input_resolution.item()
context_length = clip_model.context_length.item()
vocab_size = clip_model.vocab_size.item()

def find_clip_accuracies(model):

    # Encode text and create zero-shot classifier with prompt templates
    def zeroshot_classifier(classnames, templates):
        with torch.no_grad():
            zeroshot_weights = []
            for classname in tqdm(classnames, desc="zeroshot classifier"):
                texts = [template.format(classname) for template in templates] # format with class
                texts = clip.tokenize(texts).cuda() # tokenize
                class_embeddings = model.encode_text(texts) # embed with text encoder
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
                zeroshot_weights.append(class_embedding)
            zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
        return zeroshot_weights

    filename = "zeroshot_weights_prompts.pk"
    # zeroshot_weights = zeroshot_classifier(list(wnid_to_labels_openai.values()), imagenet_templates_openai)
    # with open(filename, 'wb') as fi:
    #     pickle.dump(zeroshot_weights, fi)
    with open(filename, 'rb') as fi:
        zeroshot_weights = pickle.load(fi)

    top1 = 0
    top5 = 0
    directory = r'/localtmp/data/imagenet256/val/'

    # Iterate over every validation class
    for wnid in tqdm(os.listdir(directory)):
        subdirectory = directory + wnid + r'/'
        images = []

        # Preprocess images
        for filename in os.listdir(subdirectory):
            filepath = subdirectory + filename
            image = Image.open(filepath).convert('RGB')
            image_input = preprocess(image).unsqueeze(0).to(device)
            images += [image_input]
        images = torch.cat(images, 0)

        # Encode images and evaluate similarities in batch sizes of 50
        with torch.no_grad():
            image_features = model.encode_image(images)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ zeroshot_weights).softmax(dim=-1)
        _, indices = similarity.topk(5)

        # Count top 1 and top 5 accuracies
        for i in range(indices.shape[0]):
            predictions = [list(wnid_to_labels_openai.values())[index] for index in indices[i]]
            if wnid_to_labels_openai[wnid] == predictions[0]:
                top1 += 1
            if wnid_to_labels_openai[wnid] in predictions:
                top5 += 1

    print ('-------------------------------------')
    print (f'Top 1 accuracy: {100.0 * top1 / 50000:.2f}%')
    print (f'Top 5 accuracy: {100.0 * top5 / 50000:.2f}%')
    print ()

print (f'CLIP with prompt templates (Paper: 63.2%)')
find_clip_accuracies(clip_model)

