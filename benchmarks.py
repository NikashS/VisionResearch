#! /usr/bin/env python3

# from CLIP.simple_tokenizer import SimpleTokenizer
from wnid_dictionaries import wnid_to_labels, wnid_to_short_labels, wnid_to_hyponyms

import os
from PIL import Image
import sys
import torch
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import urllib.request

sys.path.append(os.getcwd() + '/..')
from CLIP.simple_tokenizer import SimpleTokenizer

# Load the models
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = torch.jit.load("clip_model.pt").cuda().eval()
resnet_model = torch.jit.load("resnet_model.pt").cuda().eval()

# CLIP ViT and ResNet both have the same below attributes
input_resolution = clip_model.input_resolution.item()
context_length = clip_model.context_length.item()
vocab_size = clip_model.vocab_size.item()

# Define image preprocessing method
preprocess = Compose([
    Resize(input_resolution, interpolation=Image.BICUBIC),
    CenterCrop(input_resolution),
    ToTensor()
])

# Define a tokenize method
tokenizer = SimpleTokenizer()
def tokenize(text: str, context_length: int = 77):
    sot_token = tokenizer.encoder["<|startoftext|>"]
    eot_token = tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + tokenizer.encode(text) + [eot_token]]
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
    for i, tokens in enumerate(all_tokens):
        result[i, :len(tokens)] = torch.tensor(tokens)
    return result

def find_clip_accuracies(use_short_labels=False, use_hyponyms=False, use_resnet=False):
    wn_to_labels = wnid_to_short_labels if use_short_labels else wnid_to_labels
    wn_to_hyponyms = wnid_to_hyponyms

    # Tokenize ImageNet class labels (using hyponyms if specified)
    text_inputs = torch.cat([
        tokenize(f"a photo of a{wn_to_labels[wnid]}, a type of {(wn_to_hyponyms[wnid])}")
        if use_hyponyms else
        tokenize(f"a photo of a {wn_to_labels[wnid]}") for wnid in wn_to_labels
    ]).to(device)

    with torch.no_grad():
        text_features = resnet_model.encode_text(text_inputs) if use_resnet else clip_model.encode_text(text_inputs)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    top1 = 0
    top5 = 0
    directory = r'/localtmp/data/imagenet256/val/'

    # Iterate over every validation class
    count = 0
    for wnid in os.listdir(directory):
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
            image_features = resnet_model.encode_image(images) if use_resnet else clip_model.encode_image(images)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        _, indices = similarity.topk(5)

        # Count top 1 and top 5 accuracies
        for i in range(indices.shape[0]):
            predictions = [list(wn_to_labels.values())[index] for index in indices[i]]
            if wn_to_labels[wnid] == predictions[0]:
                top1 += 1
            if wn_to_labels[wnid] in predictions:
                top5 += 1
                
    print ('-------------------------------------')
    print (f'Top 1 accuracy: {100.0 * top1 / 50000:.2f}%')
    print (f'Top 5 accuracy: {100.0 * top5 / 50000:.2f}%')
    print ()

print (f'CLIP with simple class labels (Paper: 63.2%)')
find_clip_accuracies()

print (f'CLIP with class labels and hyponyms')
find_clip_accuracies(use_hyponyms=True)

print (f'ResNet50 with simple class labels (Paper: 59.6%)')
find_clip_accuracies(use_resnet=True)

print (f'ResNet50 with class labels and hyponyms')
find_clip_accuracies(use_hyponyms=True, use_resnet=True)