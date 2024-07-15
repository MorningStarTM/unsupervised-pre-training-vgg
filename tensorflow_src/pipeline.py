import os
from glob import glob
import numpy
import cv2
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

#read the data
def load_data(path, split=0.2):
    images = shuffle(sorted(glob(os.path.join(path, "*", "*"))))
    split_rate = int(len(images) * split)
    train, valid = train_test_split(images, test_size=split_rate, random_state=42)
    train, test = train_test_split(train, test_size=split_rate, random_state=42)
    return train, valid, test

