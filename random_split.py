import os
import glob
import pandas as pd
import numpy as np
import random
import shutil

def train_validate_test_split(df, train_percent=.8, validate_percent=.1, seed=None):
		np.random.seed(seed)
		perm = np.random.permutation(df.index)
		m = len(df.index)
		train_end = int(train_percent * m)
		validate_end = int(validate_percent * m) + train_end
		print(train_end)
		print(validate_end)
		print(m)
		train = df.iloc[perm[:train_end]]
		validate = df.iloc[perm[train_end:validate_end]]
		test = df.iloc[perm[validate_end:]]
		return train, validate, test

basepath = "/home/drswchornd/Env-semantic-seg/kariDB/src_imgs/"
imgpath = "/home/drswchornd/Env-semantic-seg/kariDB/src_imgs/"
labelpath = "/home/drswchornd/Env-semantic-seg/kariDB/src_labels/"
imgtype = ".png"
labeltype = ".png"

filenames = []
folder_img = []
folder_label = []

for entry in os.listdir(basepath):
    if os.path.isfile(os.path.join(basepath, entry)):
        filenames.append(entry)

# make sure that the filenames have a fixed order before shuffling
filenames.sort()

# shuffles the ordering of filenames (deterministic given the chosen seed)
random.seed()
random.shuffle(filenames) 

# making paired data information
for fileinfo in filenames:
		dbFileName = fileinfo.split('.', 1)
		imgFileName = imgpath + dbFileName[0] + imgtype
		xmlFileName = labelpath + dbFileName[0] + labeltype
		value_img = (imgFileName)
		value_label = (xmlFileName)
		folder_img.append(value_img)
		folder_label.append(value_label)

data_mask = pd.DataFrame({"data": folder_img, "label": folder_label})
data_mask.to_csv('data/data_label.csv', index=None)

train, validate, test = train_validate_test_split(data_mask)
train.to_csv('data/train.csv', index=None)
validate.to_csv('data/validate.csv', index=None)
test.to_csv('data/test.csv', index=None)

train_img_path = "/home/drswchornd/Env-semantic-seg/kariDB/train/"
train_label_path = "/home/drswchornd/Env-semantic-seg/kariDB/train_labels/"
for train_img in train.iloc[0:].data:
    shutil.copy(train_img, train_img_path)
for train_label in train.iloc[0:].label:
    shutil.copy(train_label, train_label_path)

test_img_path = "/home/drswchornd/Env-semantic-seg/kariDB/test/"
test_label_path = "/home/drswchornd/Env-semantic-seg/kariDB/test_labels/"
for test_img in test.iloc[0:].data:
    shutil.copy(test_img, test_img_path)
for test_label in test.iloc[0:].label:
    shutil.copy(test_label, test_label_path)

validate_img_path = "/home/drswchornd/Env-semantic-seg/kariDB/val/"
validate_label_path = "/home/drswchornd/Env-semantic-seg/kariDB/val_labels/"
for validate_img in validate.iloc[0:].data:
    shutil.copy(validate_img, validate_img_path)
for validate_label in validate.iloc[0:].label:
    shutil.copy(validate_label, validate_label_path)		
