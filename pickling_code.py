    model_name = 'resnet' if use_resnet else 'vit'

    full_prompts_file = f'zeroshot_classifier_pickles/{model_name}/full_prompts.pk'
    subset_prompts_file = f'zeroshot_classifier_pickles/{model_name}/subset_prompts.pk'
    no_prompts_file = f'zeroshot_classifier_pickles/{model_name}/no_prompts.pk'

    if use_prompts=='yes':
        with open(full_prompts_file, 'wb') as fi:
            pickle.dump(zeroshot_weights, fi)
    elif use_prompts=='subset':
        with open(subset_prompts_file, 'wb') as fi:
            pickle.dump(zeroshot_weights, fi)
    elif use_prompts=='no':
        with open(no_prompts_file, 'wb') as fi:
            pickle.dump(zeroshot_weights, fi)

    if use_prompts=='yes':
        with open(full_prompts_file, 'rb') as fi:
            zeroshot_weights = pickle.load(fi)
    elif use_prompts=='subset':
        with open(subset_prompts_file, 'rb') as fi:
            zeroshot_weights = pickle.load(fi)
    elif use_prompts=='no':
        with open(no_prompts_file, 'rb') as fi:
            zeroshot_weights = pickle.load(fi)