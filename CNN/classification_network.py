import os
import cv2
import time
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.client import device_lib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Conv2D, MaxPool2D, Flatten, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import TensorBoard

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


# check Version and if GPU is available
print("Tensorflow-Version: ", tf.__version__)
print("Device: ", device_lib.list_local_devices())
print("Keras: ", keras.__version__)


def get_X_y(IMG_DIR, get_color_img, IMG_SIZE):

    X = []
    y = []
    class_names = []

    for counter, folder_name in enumerate(os.listdir(IMG_DIR)):
        class_names.append(folder_name)

        if get_color_img:
            for img_name in tqdm(os.listdir(IMG_DIR + folder_name)):
                img_path = IMG_DIR + folder_name +"/"+ img_name

                img = cv2.imread(img_path)
                size = IMG_SIZE
                img = cv2.resize(img, (size, size))

                X.append(img/255)
                y.append(counter)
        else:
            for img_name in tqdm(os.listdir(IMG_DIR + folder_name)):
                img_path = IMG_DIR + folder_name +"/"+ img_name

                img = cv2.imread(img_path, 0)
                size = IMG_SIZE
                img = cv2.resize(img, (size, size))

                X.append(img/255)
                y.append(counter)


    print("class_names: ", class_names)
    X = np.array(X)
    y = np.array(y)
    class_names = np.array(class_names)

    X = X.astype('float64')
    y = y.astype('int32')

    return X, y, class_names

def evaluate_model(model, X_test, y_test, print_eval=True):
    prediction = np.argmax(model.predict(X_test), axis=1)
    real = y_test.reshape(-1,)
    evaluation = np.mean(prediction == real)

    if print_eval:
        print(evaluation)

    return evaluation

def save_X_y_class_names(X, y, class_names, name_postfix, colored_images=True):

    if get_color_img:
        np.save('X_color_'+ name_postfix +'.npy', X)
        np.save('y_color_'+ name_postfix +'.npy', y)
    else:
        np.save('X_gray_'+ name_postfix +'.npy', X)
        np.save('y_gray_'+ name_postfix +'.npy', y)

    np.save('class_names_'+ name_postfix +'.npy', class_names)

def load_X_y_class_names(name_postfix, colored_images=True):
    if get_color_img:
        X = np.load('X_color_'+ name_postfix +'.npy')
        y = np.load('y_color_'+ name_postfix +'.npy')
    else:
        X = np.load('X_gray_'+ name_postfix +'.npy')
        y = np.load('y_gray_'+ name_postfix +'.npy')

        X = np.expand_dims(X, axis=3)
        y = np.expand_dims(y, axis=1)

    class_names = np.load('class_names_'+ name_postfix +'.npy')

    return X, y, class_names


def model_show_predicted_images(model, X, y, class_names, colored_images=True, model_prediction=True):

    if colored_images:
        images = np.array(X*255, dtype='uint8')
        fig = plt.figure(figsize=(10, 12), dpi=200)

        for i in range(0,20):
            img_fig = fig.add_subplot(5,4,i+1)
            img_fig.axes.get_xaxis().set_visible(False)
            img_fig.axes.get_yaxis().set_visible(False)
            img_to_predict = np.array([X[i]])
            predicted_img = np.argmax(model.predict(img_to_predict), axis=1)
            plt.title(str(class_names[predicted_img[0]]))
            img_fig.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))

    else:
        images = np.array(X*255, dtype='uint8')
        fig = plt.figure(figsize=(10, 12), dpi=200)

        for i in range(0,20):
            img_fig = fig.add_subplot(5,4,i+1)
            img_fig.axes.get_xaxis().set_visible(False)
            img_fig.axes.get_yaxis().set_visible(False)
            img_to_predict = np.array([X[i]])
            predicted_img = np.argmax(model.predict(img_to_predict), axis=1)
            plt.title(str(class_names[predicted_img[0]]))
            img_fig.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
