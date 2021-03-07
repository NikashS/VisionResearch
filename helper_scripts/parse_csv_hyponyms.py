import csv
import os
import sys
from tqdm import tqdm

sys.path.append(os.getcwd() + '/..')
from wnid_dictionaries import wnid_to_short_hyponyms

def parse_csv_hyponyms():
    all_responses = {}
    best_responses = {}
    all_responses_no_duplicates = {}
    best_responses_no_duplicates = {}
    all_responses_joined = {}
    best_responses_two = {}
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
        all_responses_joined[wnid] = ', '.join(list(set(all_responses[wnid])))
        all_responses_no_duplicates[wnid] = [answer for answer in all_responses[wnid] if answer.lower() != wnid_to_short_hyponyms[wnid].lower()]
        if len(all_responses_no_duplicates[wnid]) == 0:
            all_responses_no_duplicates[wnid] = [wnid_to_short_hyponyms[wnid]]
        best_responses_no_duplicates[wnid] = max(set(all_responses_no_duplicates[wnid]), key=all_responses[wnid].count)
        sorted_responses = sorted(set(all_responses_no_duplicates[wnid]), key=all_responses_no_duplicates[wnid].count, reverse=True)
        if len(sorted_responses) > 1:
            best_responses_two[wnid] = sorted_responses[0] + ' or ' + sorted_responses[1]
        else:
            best_responses_two[wnid] = sorted_responses[0]
    return (best_responses, all_responses_joined, best_responses_no_duplicates, best_responses_two)
    
    # print (best_responses)

wnid_to_categories, wnid_to_multiple_categories, wnid_to_different_categories, wnid_to_two_categories = parse_csv_hyponyms()
