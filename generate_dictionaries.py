import urllib.request
import os

all_hyponyms = {}
path = '/u/lab/ns6td/research/hyponyms.txt'
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
    for wnid in os.listdir(directory):
        label_long = get_label_from_wnid(wnid)
        label = label_long[:label_long.find(',')]
        wnids_long[wnid] = label_long
        wnids[wnid] = label
        
    for wnid in wnids:
        hyponym = get_label_from_wnid(all_hyponyms[wnid])
        hyponyms[wnid] = hyponym
    print (hyponyms)

make_wnids_dicts()