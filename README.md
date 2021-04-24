# Zero-shot learning with CLIP

4th year thesis research project with Professor Vicente Ordonez-Roman. For my project, I improve the zero-shot learning accuracy of CLIP, an OpenAI model that produces state-of-the-art accuracies on a wide range of datasets. I also implement a novel zero-shot learning approach for predicting a classifier for unseen classes using Wikipedia textual descriptions of classes. Read more about CLIP [here](https://openai.com/blog/clip/) and view it's source code [here](https://github.com/openai/CLIP).

## Getting started

Clone this repository and the CLIP repository linked above in the same directory. Execute `pip install -r requirements.txt` in your virtualenv.

Running this code requires access to UVA's downloaded ImageNet dataset, but this can be easily replaced with ImageNetV2 which is publically available online. Additionally, the python scripts should be run using a CUDA GPU environment.

To improve model loading performance, optionally download and load the CLIP ViT model from [here](https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt), and save it into a `.pt` file.

## Overview of repository

`prompt_engineering.py` is the main python script. This file benchmarks CLIP zero-shot learning with prompt engineering and hyponyms on the ImageNet dataset and achieves an accuracy score of 65.8% accuracy (compared to OpenAI's 63.2% accuracy in `first_benchmarks.py`).

`prompt_templates.py` contains caption templates for images to improve CLIP's accuracy, and these templates are used in `prompt_engineering.py`.

### CUB Predicting Encodings

Inside the `cub` directory is a zero-shot learning approach using CLIP for the Caltech-UCSD Birds (CUB) dataset. This code learns the weights of a logistic regression classifier trained on seen classes of the CUB dataset, and trains a multi-layer perceptron on wikipedia descriptions of each bird class to predict weights for unseen classes.

`predicting_encodings.py` is the main python script here. Download the CUB dataset [here](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) and corresponding Wikipedia textual descriptions [here](http://deep.cs.virginia.edu/data/cub/birds_wikipedia/)
