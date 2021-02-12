print (f'CLIP ViT with prompt templates (Paper: 63.2%)')
find_clip_accuracies(use_prompts='yes')

print (f'CLIP ViT without prompt templates')
find_clip_accuracies(use_prompts='no')

print (f'CLIP ViT with subset of prompt templates')
find_clip_accuracies(use_prompts='subset')

print (f'CLIP ViT with best (not ensembled) prompt template')
find_clip_accuracies(use_prompts='yes', ensemble_prompts=False)

print (f'CLIP ViT with best (not ensembled) prompt template with subset')
find_clip_accuracies(use_prompts='subset', ensemble_prompts=False)

print (f'CLIP ViT with prompt templates and with hyponyms')
find_clip_accuracies(use_prompts='yes', use_hyponyms=True)

print (f'CLIP ViT without prompt templates and with hyponyms')
find_clip_accuracies(use_prompts='no', use_hyponyms=True)

print (f'CLIP ViT with subset prompt templates and with hyponyms')
find_clip_accuracies(use_prompts='subset', use_hyponyms=True)

print (f'CLIP ViT with prompt templates (standard ImageNet classes)')
find_clip_accuracies(use_prompts='yes', use_openai_imagenet_classes=True)

print (f'CLIP ViT without prompt templates (standard ImageNet classes)')
find_clip_accuracies(use_prompts='no', use_openai_imagenet_classes=True)

print (f'CLIP ViT with subset of prompt templates (standard ImageNet classes)')
find_clip_accuracies(use_prompts='subset', use_openai_imagenet_classes=True)

print (f'CLIP ViT with prompt templates and with hyponyms (standard ImageNet classes)')
find_clip_accuracies(use_prompts='yes', use_hyponyms=True, use_openai_imagenet_classes=True)

print (f'CLIP ViT without prompt templates and with hyponyms (standard ImageNet classes)')
find_clip_accuracies(use_prompts='no', use_hyponyms=True, use_openai_imagenet_classes=True)

print (f'CLIP ViT with subset prompt templates and with hyponyms (standard ImageNet classes)')
find_clip_accuracies(use_prompts='subset', use_hyponyms=True, use_openai_imagenet_classes=True)

print (f'CLIP ViT with prompt templates and with common hierarchal hyponyms')
find_clip_accuracies(use_prompts='yes', use_hyponyms=True, hyponyms_dict=common_hyponyms)

print (f'CLIP ViT without prompt templates and with common hierarchal hyponyms')
find_clip_accuracies(use_prompts='no', use_hyponyms=True, hyponyms_dict=common_hyponyms)

print (f'CLIP ViT with subset prompt templates and with common hierarchal hyponyms')
find_clip_accuracies(use_prompts='subset', use_hyponyms=True, hyponyms_dict=common_hyponyms)

print (f'CLIP ViT with prompt templates and with human generated hyponyms')
find_clip_accuracies(use_prompts='yes', use_hyponyms=True, hyponyms_dict=common_hyponyms_human)

print (f'CLIP ViT without prompt templates and with human generated hyponyms')
find_clip_accuracies(use_prompts='no', use_hyponyms=True, hyponyms_dict=common_hyponyms_human)

print (f'CLIP ViT with subset prompt templates and with human generated hyponyms')
find_clip_accuracies(use_prompts='subset', use_hyponyms=True, hyponyms_dict=common_hyponyms_human)