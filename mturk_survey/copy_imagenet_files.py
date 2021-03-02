import os
from shutil import copy2
import sys
from tqdm import tqdm

def copy():
    directory = r'/localtmp/data/imagenet256/train/'
    destination = r'/u/lab/ns6td/public_html/data/train/'
    for wnid in tqdm(os.listdir(directory), desc="wnids dict"):
        os.makedirs(destination + wnid, exist_ok=True)
        subdirectory = directory + wnid + r'/'
        for image in os.listdir(subdirectory)[:10]:
            image_path = subdirectory + image
            copy2(image_path, destination + wnid + r'/')

copy()