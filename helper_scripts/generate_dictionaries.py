import urllib.request
import os
from tqdm import tqdm
from wnid_dictionaries import wnid_to_short_hyponyms

all_hyponyms = {}
path = '/u/lab/ns6td/VisionResearch/imagenet_hyponyms.txt'
associations_file = open(path,'r')
text = associations_file.read()
associations = text.split('\n')[:-1]
for association in associations:
    pair = association.split(' ')
    parent = pair[0]
    child = pair[1]
    all_hyponyms[child] = parent

def get_label_from_wnid(wnid):
    with urllib.request.urlopen('http://www.image-net.org/api/text/wordnet.synset.getwords?wnid=' + wnid) as response:
        result = response.read().decode("utf-8")[:-1].replace('\n', ', ')
        return result

def make_wnids_dicts():
    wnids = {}
    wnids_long = {}
    hyponyms = {}
    directory = r'/localtmp/data/imagenet256/val/'
    # Iterate over every validation class
    for wnid in tqdm(os.listdir(directory), desc="wnids dict"):
        label_long = get_label_from_wnid(wnid)
        label = label_long[:label_long.find(',')]
        wnids_long[wnid] = label_long
        wnids[wnid] = label
        
    for wnid in wnids:
        hyponym = get_label_from_wnid(all_hyponyms[wnid])
        hyponyms[wnid] = hyponym
    return wnids

def make_common_hyponyms_dicts():
    val_wnids = set([])
    common_hyponyms = wnid_to_short_hyponyms.copy()
    common_path = '/u/lab/ns6td/VisionResearch/common_hyponyms.txt'
    human_path = '/u/lab/ns6td/VisionResearch/common_human_hyponyms.txt'
    common_pairs = open(common_path, 'r').read().split('\n')
    human_pairs = open(human_path, 'r').read().split('\n')
    directory = r'/localtmp/data/imagenet256/val/'
    for pair in common_pairs:
        wnid = pair[:pair.find(' ')]
        text = pair[pair.find(' ') + 1:]
        if wnid in common_hyponyms:
            common_hyponyms[wnid] = text
    common_hyponyms_human = common_hyponyms.copy()
    count = 0
    for pair in human_pairs:
        wnid = pair[:pair.find(' ')]
        text = pair[pair.find(' ') + 1:]
        if wnid in common_hyponyms_human and text != common_hyponyms_human[wnid]:
            common_hyponyms_human[wnid] = text
            count += 1
    # print (common_hyponyms)
    print (common_hyponyms_human)

make_common_hyponyms_dicts()