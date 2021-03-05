import numpy as np
import os
from PIL import Image
import sys
import torch
from tqdm import tqdm

print ("starting...")

sys.path.append(os.getcwd() + '/..')
from CLIP import clip
from prompt_templates import prompt_templates_openai, subset_prompt_templates_openai
from wnid_dictionaries import wnid_to_labels, wnid_to_categories, wnid_to_labels_openai, wnid_to_short_hyponyms, common_hyponyms, common_hyponyms_human

device = 'cuda' if torch.cuda.is_available() else 'cpu'
clip_model, preprocess = clip.load('ViT-B/32')
resnet_model, _ = clip.load('RN50')

def find_clip_accuracies(
    use_resnet=False,
    use_prompts='yes',
    ensemble_prompts=True,
    use_hyponyms=False,
    use_openai_imagenet_classes=True,
    hyponyms_dict=wnid_to_short_hyponyms,
    hyponym_template=', a type of {}',
):
    if use_prompts=='yes':
        prompt_templates = prompt_templates_openai
    elif use_prompts=='subset':
        prompt_templates = subset_prompt_templates_openai
    elif use_prompts=='no':
        prompt_templates = ['a photo of a {}.']
    # Encode text and create zero-shot classifier with prompt templates
    def zeroshot_classifier(classes):
        with torch.no_grad():
            zeroshot_weights = []
            """ for wnid in tqdm(classes, desc='generate text classifier', leave=False): """
            for wnid in classes:
                label = f'{classes[wnid]}{hyponym_template.format(hyponyms_dict[wnid])}' if use_hyponyms else classes[wnid]
                texts = [template.format(label+' ') if template[-2] != '}' else template.format(label) for template in prompt_templates]
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
    directory = r'/localtmp/data/imagenet256/train/'

    # Iterate over first 50 images in each train class
    for wnid in tqdm(os.listdir(directory), desc='find zeroshot accuracy', leave=False):
    # for wnid in os.listdir(directory):
        subdirectory = directory + wnid + r'/'
        images = []

        # Preprocess images
        for filename in os.listdir(subdirectory)[:50]:
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
            divisor = 1 if ensemble_prompts else len(prompt_templates)
            predictions = [list(wnid_to_labels_openai.values())[index // divisor] for index in indices[i]]
            if wnid_to_labels_openai[wnid] == predictions[0]:
                top1 += 1
            if wnid_to_labels_openai[wnid] in predictions:
                top5 += 1

    print ('-'*40)
    print (f'Top 1 accuracy: {100.0 * top1 / 50000:.2f}%')
    if use_prompts=="no" or ensemble_prompts:
        print (f'Top 5 accuracy: {100.0 * top5 / 50000:.2f}%')
    print ()

prompts_options = [('subset', 'with subset of')]
for prompts_option, benchmark_title in prompts_options:

    hyponym_templates = [', which is a type of {}']
    for hyponym_template in hyponym_templates:

        print (f'CLIP ViT {benchmark_title} prompt templates and with mturk survey results (using {hyponym_template})')
        find_clip_accuracies(use_prompts=prompts_option, use_hyponyms=True, hyponyms_dict=wnid_to_categories, hyponym_template=hyponym_template)

    print (f'CLIP ViT {benchmark_title} prompt templates')
    find_clip_accuracies(use_prompts=prompts_option)