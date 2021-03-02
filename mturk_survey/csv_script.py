import csv
from lxml import html
import os
import random
import requests
import sys
from tqdm import tqdm

def write_csv():
    filename = '/u/lab/ns6td/VisionResearch/mturk_survey/image_urls.csv'
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        header_list = ['wnid', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten']
        five = ['one_', 'two_', 'three_', 'four_', 'five_']
        writer.writerow([prepend + header for prepend in five for header in header_list])

        counter = 0
        row = []
        directory = r'/u/lab/ns6td/public_html/data/train/'
        base_url = 'http://www.cs.virginia.edu/~ns6td/data/train/'
        for wnid in tqdm(os.listdir(directory)):
            counter += 1
            subdirectory = directory + wnid + '/'
            urls = [base_url + wnid + '/' + image_name for image_name in os.listdir(subdirectory)]
            csv_data = [wnid] + urls
            row = row + csv_data
            if counter == 5:
                writer.writerow(row)
                row = []
                counter = 0
        return

write_csv()