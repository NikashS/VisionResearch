# Zero-shot learning with CLIP

4th year thesis research project with Professor Vicente Ordonez-Roman. For my project, I will be improving the zero-shot learning accuracy of CLIP, an OpenAI model that produces state-of-the-art accuracies on a wide range of datasets. Read more about CLIP [here](https://openai.com/blog/clip/) and view it's source code [here](https://github.com/openai/CLIP).

## Getting started

Clone this repository and the CLIP repository linked above in the same directory. 

Running this code requires access to UVA's downloaded ImageNet dataset, but this can be easily replaced with ImageNetV2 which is publically available online. Additionally, the python scripts should be run using a CUDA GPU environment.

To improve model loading performance, optionally download and load the CLIP ViT model from [here](https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt), and save it into a `.pt` file.

## Overview of repository

`prompt_engineering.py` (benchmarks of CLIP zero-shot learning with prompt engineering and hyponyms) and `linear_probe_prompt_engineering.py` are the main python scripts.

`generate_dictionaries.py` is a one-time script to produce the dictionaries in `wnid_dictionaries.py`.

`prompt_templates.py` contains caption templates for images to improve CLIP's accuracy, and these templates are used in `prompt_engineering.py`.
