import csv
import os
from tqdm import tqdm

def parse_csv_hyponyms():
    all_responses = {}
    best_responses = {}
    all_responses_joined = {}
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
    for wnid in all_responses:
        best_responses[wnid] = max(set(all_responses[wnid]), key=all_responses[wnid].count)
    for wnid in all_responses:
        all_responses_joined[wnid] = ', '.join(list(set(all_responses[wnid])))
    return (best_responses, all_responses_joined)
    
    # print (best_responses)

wnid_to_categories, wnid_to_multiple_categories = parse_csv_hyponyms()
