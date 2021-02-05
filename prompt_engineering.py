import numpy as np
import os
import pickle
from PIL import Image
import sys
import torch
from tqdm import tqdm

sys.path.append(os.getcwd() + '/..')
from CLIP import clip
from prompt_templates import prompt_templates_openai, subset_prompt_templates_openai
from wnid_dictionaries import wnid_to_labels_openai, wnid_to_hyponyms

device = 'cuda' if torch.cuda.is_available() else 'cpu'
clip_model, preprocess = clip.load('ViT-B/32')
resnet_model, _ = clip.load('RN50')

# Model parameters of ViT and RN are the same
input_resolution = clip_model.input_resolution.item()
context_length = clip_model.context_length.item()
vocab_size = clip_model.vocab_size.item()

def find_clip_accuracies(use_resnet=False, use_prompts='yes', ensemble_prompts=True, use_hyponyms=False):

    # Encode text and create zero-shot classifier with prompt templates
    def zeroshot_classifier(classes):
        with torch.no_grad():
            zeroshot_weights = []
            for wnid in tqdm(classes, desc='zeroshot classifier', leave=False):
                label = f'{classes[wnid]} (a type of {wnid_to_hyponyms[wnid]})' if use_hyponyms else classes[wnid]
                if use_prompts=='yes':
                    texts = [template.format(label) for template in prompt_templates_openai]
                elif use_prompts=='subset':
                    texts = [template.format(label) for template in subset_prompt_templates_openai]
                elif use_prompts=='no':
                    texts = f'a photo of a {label}.'
                texts = clip.tokenize(texts).cuda()
                class_embeddings = resnet_model.encode_text(texts) if use_resnet else clip_model.encode_text(texts)
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
                if ensemble_prompts:
                    zeroshot_weights.append(class_embedding)
                else:
                    for embedding in class_embeddings:
                        zeroshot_weights.append(embedding)
            zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
        return zeroshot_weights

    zeroshot_weights = zeroshot_classifier(wnid_to_labels_openai)

    top1 = 0
    top5 = 0
    directory = r'/localtmp/data/imagenet256/val/'

    # Iterate over every validation class
    for wnid in tqdm(os.listdir(directory), desc='zeroshot classifier', leave=False):
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
        similarity = (100.0 * image_features @ zeroshot_weights).softmax(dim=-1)
        _, indices = similarity.topk(5)

        # Count top 1 and top 5 accuracies
        for i in range(indices.shape[0]):
            divisor = 1
            if not ensemble_prompts:
                if use_prompts=='yes':
                    divisor = len(prompt_templates_openai)
                elif use_prompts=='subset':
                    divisor = len(subset_prompt_templates_openai)
            predictions = [list(wnid_to_labels_openai.values())[index // divisor] for index in indices[i]]
            if wnid_to_labels_openai[wnid] == predictions[0]:
                top1 += 1
            if wnid_to_labels_openai[wnid] in predictions:
                top5 += 1

    print ('-------------------------------------')
    print (f'Top 1 accuracy: {100.0 * top1 / 50000:.2f}%')
    print (f'Top 5 accuracy: {100.0 * top5 / 50000:.2f}%') if not ensemble_prompts
    print ()

print (f'CLIP ViT with prompt templates (Paper: 63.2%)')
find_clip_accuracies(use_resnet=False, use_prompts='yes')

print (f'ResNet with prompt templates (Paper: 59.6%)')
find_clip_accuracies(use_resnet=True, use_prompts='yes')

print (f'CLIP ViT without prompt templates')
find_clip_accuracies(use_resnet=False, use_prompts='no')

print (f'ResNet without prompt templates')
find_clip_accuracies(use_resnet=True, use_prompts='no')

print (f'CLIP ViT with subset of prompt templates')
find_clip_accuracies(use_resnet=False, use_prompts='subset')

print (f'ResNet with subset of prompt templates')
find_clip_accuracies(use_resnet=True, use_prompts='subset')

print (f'CLIP ViT with best (not ensembled) prompt template')
find_clip_accuracies(use_resnet=False, use_prompts='yes', ensemble_prompts=False)

print (f'ResNet with best (not ensembled) prompt template')
find_clip_accuracies(use_resnet=True, use_prompts='yes', ensemble_prompts=False)

print (f'CLIP ViT with best (not ensembled) prompt template with subset')
find_clip_accuracies(use_resnet=False, use_prompts='subset', ensemble_prompts=False)

print (f'ResNet with best (not ensembled) prompt template with subset')
find_clip_accuracies(use_resnet=True, use_prompts='subset', ensemble_prompts=False)

print (f'CLIP ViT with prompt templates and with hyponyms')
find_clip_accuracies(use_resnet=False, use_prompts='yes', use_hyponyms=True)

print (f'ResNet with prompt templates and with hyponyms')
find_clip_accuracies(use_resnet=True, use_prompts='yes', use_hyponyms=True)

print (f'CLIP ViT without prompt templates and with hyponyms')
find_clip_accuracies(use_resnet=False, use_prompts='no', use_hyponyms=True)

print (f'ResNet without prompt templates and with hyponyms')
find_clip_accuracies(use_resnet=True, use_prompts='no', use_hyponyms=True)

print (f'CLIP ViT with subset prompt templates and with hyponyms')
find_clip_accuracies(use_resnet=False, use_prompts='subset', use_hyponyms=True)

print (f'ResNet with subset prompt templates and with hyponyms')
find_clip_accuracies(use_resnet=True, use_prompts='subset', use_hyponyms=True)

