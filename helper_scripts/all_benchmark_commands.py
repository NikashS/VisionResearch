prompts_options = [('yes', 'with'), ('subset', 'with subset of'), ('no', 'without')]
for prompts_option, benchmark_title in prompts_options:

    print (f'CLIP ViT {benchmark_title} prompt templates')
    find_clip_accuracies(use_prompts=prompts_option)

    hyponym_templates = [', a type of {}', ', which is a type of {}']
    for hyponym_template in hyponym_templates:

        print (f'CLIP ViT {benchmark_title} prompt templates and with ImageNet hyponyms (using {hyponym_template})')
        find_clip_accuracies(use_prompts=prompts_option, use_hyponyms=True, hyponym_template=hyponym_template)

        print (f'CLIP ViT {benchmark_title} prompt templates and with common hierarchal hyponyms (using {hyponym_template})')
        find_clip_accuracies(use_prompts=prompts_option, use_hyponyms=True, hyponyms_dict=common_hyponyms, hyponym_template=hyponym_template)

        print (f'CLIP ViT {benchmark_title} prompt templates and with human generated hyponyms (using {hyponym_template})')
        find_clip_accuracies(use_prompts=prompts_option, use_hyponyms=True, hyponyms_dict=common_hyponyms_human, hyponym_template=hyponym_template)

print (f'CLIP ViT with best (not ensembled) prompt template')
find_clip_accuracies(use_prompts='yes', ensemble_prompts=False)

print (f'CLIP ViT with best (not ensembled) prompt template with subset')
find_clip_accuracies(use_prompts='subset', ensemble_prompts=False)