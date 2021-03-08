import numpy as np
import os
from PIL import Image
import sys
import torch
from tqdm import tqdm

print ("starting...")

sys.path.append(os.getcwd() + '/..')
sys.path.append(os.getcwd() + '/../..')
from CLIP import clip
from prompt_templates import prompt_templates_openai, subset_prompt_templates_openai
from cub_resources.id_to_labels import id_to_labels, id_to_top_attributes

device = 'cuda' if torch.cuda.is_available() else 'cpu'
clip_model, preprocess = clip.load('ViT-B/32')

def find_clip_accuracies(
    use_prompts='yes',
    num_attributes=0,
    num_ensembles=1,
):
    if use_prompts=='yes':
        prompt_templates = prompt_templates_openai
    elif use_prompts=='subset':
        prompt_templates = subset_prompt_templates_openai
    elif use_prompts=='no':
        prompt_templates = ['a photo of a {}.']
    elif use_prompts=='bird':
        prompt_templates = ['a photo of a {}, a type of bird.']
    # Encode text and create zero-shot classifier with prompt templates
    def zeroshot_classifier(classes):
        with torch.no_grad():
            zeroshot_weights = []
            for class_id in classes:
                texts = []
                for ensemble in range(num_ensembles):
                    label = id_to_labels[class_id]
                    if num_attributes > 0:
                        label += ', which ' + ', '.join(id_to_top_attributes[class_id][num_attributes*ensemble:num_attributes*(ensemble+1)])
                    for template in prompt_templates:
                        caption = template.format(label+',') if template[-2] != '}' else template.format(label)
                        texts.append(caption)
                texts = clip.tokenize(texts).cuda()
                class_embeddings = clip_model.encode_text(texts)
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
                zeroshot_weights.append(class_embedding)
            zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
        return zeroshot_weights

    zeroshot_weights = zeroshot_classifier(id_to_labels)

    top1 = 0
    top5 = 0
    directory = r'/localtmp/data/cub/CUB_200_2011/images/'

    # Iterate over first 50 images in each train class
    for id_and_classname in tqdm(os.listdir(directory), desc='find zeroshot accuracy', leave=False):
    # for wnid in os.listdir(directory):
        class_id = id_and_classname.split('.')[0]
        subdirectory = directory + id_and_classname + r'/'
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
            image_features = clip_model.encode_image(images)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ zeroshot_weights).softmax(dim=-1)
        _, indices = similarity.topk(5)

        # Count top 1 and top 5 accuracies
        for i in range(indices.shape[0]):
            predictions = [list(id_to_labels.values())[index] for index in indices[i]]
            if id_to_labels[int(class_id)] == predictions[0]:
                top1 += 1
            if id_to_labels[int(class_id)] in predictions:
                top5 += 1

    print ('-'*40)
    print (f'Accuracy: {100.0 * top1 / 10000:.2f}%')
    print ()

prompts_options = [('no', 'without'), ('bird', 'with the bird')]
for prompts_option, benchmark_title in prompts_options:

    print (f'CLIP ViT ensembling 5 prompts of best attribute and {benchmark_title} prompt templates')
    find_clip_accuracies(use_prompts=prompts_option, num_attributes=1, num_ensembles=5)

    print (f'CLIP ViT ensembling 5 prompts of top 3 attributes and {benchmark_title} prompt templates')
    find_clip_accuracies(use_prompts=prompts_option, num_attributes=3, num_ensembles=5)

    print (f'CLIP ViT with top 3 attribute and {benchmark_title} prompt templates')
    find_clip_accuracies(use_prompts=prompts_option, num_attributes=3)

    print (f'CLIP ViT with best attribute and {benchmark_title} prompt templates')
    find_clip_accuracies(use_prompts=prompts_option, num_attributes=1)

    print (f'CLIP ViT {benchmark_title} prompt templates')
    find_clip_accuracies(use_prompts=prompts_option)