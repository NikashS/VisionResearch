import os
import sys

id_to_labels = {}

path = '/localtmp/data/cub/CUB_200_2011/classes.txt'
classes_file = open(path, 'r')
text = classes_file.read()
lines = text.split('\n')[:-1]
for line in lines:
    pair = line.split(' ')
    class_id = pair[1][0:3]
    class_name = pair[1][4:].replace('_', ' ')
    id_to_labels[class_id] = class_name