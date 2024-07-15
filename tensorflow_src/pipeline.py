import os
from glob import glob
import numpy
import cv2
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt


#read the data
def load_data(path, split=0.2):
    images = shuffle(sorted(glob(os.path.join(path, "*", "*"))))
    split_rate = int(len(images) * split)
    train, valid = train_test_split(images, test_size=split_rate, random_state=42)
    train, test = train_test_split(train, test_size=split_rate, random_state=42)
    return train, valid, test


#create folder for save augmented images
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def PlotImg(img_arr):
    fig, axes = plt.subplots(1, 10, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip(img_arr, axes):
        ax.imshow(img)
        ax.axis('off')

    plt.tight_layout()
    plt.show()