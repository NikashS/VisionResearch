import csv
from lxml import html
import os
import random
import requests
import sys
from tqdm import tqdm

sys.path.append(os.getcwd() + '/..')
from wnid_dictionaries import wnid_to_labels_openai

def write_csv():
    filename = '/u/lab/ns6td/VisionResearch/mturk_survey/image_urls.csv'
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['wnid', 'name', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten'])

        for wnid in tqdm(wnid_to_labels_openai):
            class_name = wnid_to_labels_openai[wnid]
            url = 'http://deep.cs.virginia.edu/data/imagenet256/train/' + wnid + '/'
            page = requests.get(url)
            tree = html.fromstring(page.content)
            picture_names = tree.xpath('//a/text()')
            urls = [url + picture_name for picture_name in picture_names[20:30]]
            csv_data = [wnid, class_name] + urls
            writer.writerow(csv_data)
        return

write_csv()