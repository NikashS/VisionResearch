import numpy as np
import os
from PIL import Image
import sys
import torch
from tqdm import tqdm

sys.path.append(os.getcwd() + '/..')
from CLIP import clip
from prompt_templates import prompt_templates_openai, subset_prompt_templates_openai
from wnid_dictionaries import wnid_to_labels, wnid_to_labels_openai, wnid_to_short_hyponyms

device = 'cuda' if torch.cuda.is_available() else 'cpu'
clip_model, preprocess = clip.load('ViT-B/32')
resnet_model, _ = clip.load('RN50')

def find_clip_accuracies(
    use_resnet=False,
    use_prompts='yes',
    ensemble_prompts=True,
    use_hyponyms=False,
    use_openai_imagenet_classes=True,
):
    # Encode text and create zero-shot classifier with prompt templates
    def zeroshot_classifier(classes):
        with torch.no_grad():
            zeroshot_weights = []
            for wnid in tqdm(classes, desc='generate text classifier', leave=False):
                label = f'{classes[wnid]} (a type of {wnid_to_short_hyponyms[wnid]})' if use_hyponyms else classes[wnid]
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

    if use_openai_imagenet_classes:
        zeroshot_weights = zeroshot_classifier(wnid_to_labels_openai)
    else:
        zeroshot_weights = zeroshot_classifier(wnid_to_labels)

    top1 = 0
    top5 = 0
    directory = r'/localtmp/data/imagenet256/val/'

    # Iterate over every validation class
    for wnid in tqdm(os.listdir(directory), desc='find zeroshot accuracy', leave=False):
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
    if use_prompts=="no" or ensemble_prompts:
        print (f'Top 5 accuracy: {100.0 * top5 / 50000:.2f}%')
    print ()

print (f'CLIP ViT with subset of prompt templates')
find_clip_accuracies(use_prompts='subset')

print (f'CLIP ViT with subset prompt templates and with hyponyms')
find_clip_accuracies(use_prompts='subset', use_hyponyms=True)
