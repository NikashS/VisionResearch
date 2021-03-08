import numpy
import os
import sys

from attributes_dict import attributes_dict

id_to_labels = {}

path = '/localtmp/data/cub/CUB_200_2011/classes.txt'
classes_file = open(path, 'r')
text = classes_file.read()
lines = text.split('\n')[:-1]
for line in lines:
    pair = line.split(' ')
    class_id = pair[1][0:3]
    class_name = pair[1][4:].replace('_', ' ')
    id_to_labels[int(class_id)] = class_name

counter = 1
id_to_top_attributes = {}
percentages_path = '/localtmp/data/cub/CUB_200_2011/attributes/class_attribute_labels_continuous.txt'
percentages_file = open(percentages_path, 'r')
percentages_text = percentages_file.read()
percentages_lines = percentages_text.split('\n')[:-1]
for line in percentages_lines:
    percentages = [float(percent) for percent in line.split(' ')]
    top_sorted_indices = numpy.argsort(percentages)[::-1][:15]
    top_attributes = [attributes_dict[index+1] for index in top_sorted_indices]
    id_to_top_attributes[counter] = top_attributes
    counter += 1