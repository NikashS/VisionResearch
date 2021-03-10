import csv
import os
import sys
from tqdm import tqdm

sys.path.append(os.getcwd() + '/..')
from wnid_dictionaries import wnid_to_labels_openai

def parse_csv_hyponyms():
    all_responses = {}
    all_unique_responses = {}
    best_responses = {}
    filename = r'/u/lab/ns6td/VisionResearch/data_files/mturk_hyponyms_stripped.csv'
    with open(filename, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        headers = next(reader)
        wnids = [header[7:] for header in headers]
        for wnid in wnids:
            all_responses[wnid] = []
        for row in reader:
            for i in range(len(wnids)):
                answers = [answer.strip().replace('\'', '') for answer in row[i].lower().split(',')]
                all_responses[wnids[i]] += answers
    no_uniques = set([])
    for wnid in all_responses:
        sorted_list = sorted(set(all_responses[wnid]), key=all_responses[wnid].count, reverse=True)
        all_responses[wnid] = sorted_list[:5]
        all_unique_responses[wnid] = [answer for answer in all_responses[wnid] if answer.lower() != wnid_to_labels_openai[wnid].lower()]
        if len(all_unique_responses[wnid]) == 0:
            all_unique_responses[wnid] = [wnid_to_labels_openai[wnid]]
            no_uniques.add(wnid_to_labels_openai[wnid])
        best_responses[wnid] = all_unique_responses[wnid][0]
    return (all_responses, all_unique_responses, best_responses)
    
wnid_to_categories, wnid_to_unique_categories, wnid_to_best_category = parse_csv_hyponyms()
