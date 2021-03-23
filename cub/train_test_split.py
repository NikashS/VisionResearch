import os
import sys

image_filename = '/localtmp/data/cub/CUB_200_2011/images.txt'
train_test_filename = '/localtmp/data/cub/CUB_200_2011/train_test_split.txt'

id_to_filenames = {}
image_file = open(image_filename, 'r')
image_text = image_file.read()
image = image_text.split('\n')[:-1]
for img in image:
    num, long_filename = tuple(img.split(' '))
    filename = long_filename.split('/')[1]
    id_to_filenames[num] = filename

train_filenames = set([])
test_filenames = set([])

train_test_file = open(train_test_filename, 'r')
train_test_text = train_test_file.read()
train_test = train_test_text.split('\n')[:-1]
for img in train_test:
    num, split = tuple(img.split(' '))
    if split == '1':
        train_filenames.add(id_to_filenames[num])
    else:
        test_filenames.add(id_to_filenames[num])