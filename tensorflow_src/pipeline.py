import os
from glob import glob
import numpy
import cv2
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import numpy as np
from .const import *
import tensorflow as tf


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


def process_image(path):
    #decode the path
    path = path.decode()
    #read image
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    #resize the image
    image = cv2.resize(image, IMAGE_SIZE)
    #scale the image
    image = image / 255.0
    #change the data type of image
    image = image.astype(np.float32)

    #labeling the image
    class_name = path.split("/")[-2]
    class_idx = class_names.index(class_name)
    class_idx = np.array(class_idx, dtype=np.int32)

    return image, class_idx


def parse(path):
    image, labels = tf.numpy_function(process_image, [path], (tf.float32, tf.int32))
    labels = tf.one_hot(labels, num_class)
    image.set_shape([H, W, C])
    labels.set_shape(num_class)
  
    return image, labels


#tensorflow dataset
def tf_dataset(images, batch=8):
    dataset = tf.data.Dataset.from_tensor_slices((images))
    dataset = dataset.map(parse)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(8)
    return dataset