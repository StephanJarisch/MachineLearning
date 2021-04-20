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
from tensorflow.keras.layers import GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam
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

    if not os.path.exists("data"):
        os.mkdir("data")

    if get_color_img:
        np.save('data/X_color_'+ name_postfix +'.npy', X)
        np.save('data/y_color_'+ name_postfix +'.npy', y)
    else:
        np.save('data/X_gray_'+ name_postfix +'.npy', X)
        np.save('data/y_gray_'+ name_postfix +'.npy', y)

    np.save('data/class_names_'+ name_postfix +'.npy', class_names)

def load_X_y_class_names(name_postfix, colored_images=True):
    if get_color_img:
        X = np.load('data/X_color_'+ name_postfix +'.npy')
        y = np.load('data/y_color_'+ name_postfix +'.npy')
    else:
        X = np.load('data/X_gray_'+ name_postfix +'.npy')
        y = np.load('data/y_gray_'+ name_postfix +'.npy')

        X = np.expand_dims(X, axis=3)
        y = np.expand_dims(y, axis=1)

    class_names = np.load('data/class_names_'+ name_postfix +'.npy')

    return X, y, class_names


def model_show_predicted_images(model, X, y, class_names, colored_images=True, model_prediction=True):

    if colored_images:
        images = np.array(X*255, dtype='uint8')
        fig = plt.figure(figsize=(12, 12), dpi=200)

        for i in range(0,20):
            img_fig = fig.add_subplot(5,4,i+1)
            img_fig.axes.get_xaxis().set_visible(False)
            img_fig.axes.get_yaxis().set_visible(False)
            img_to_predict = np.array([X[i]])
            predicted_img = np.argmax(model.predict(img_to_predict), axis=1)
            plt.title(str(class_names[predicted_img[0]]))
            img_fig.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))

        plt.savefig('predictions.png')
        plt.show()


    else:
        images = np.array(X*255, dtype='uint8')
        fig = plt.figure(figsize=(12, 12), dpi=200)

        for i in range(0,20):
            img_fig = fig.add_subplot(5,4,i+1)
            img_fig.axes.get_xaxis().set_visible(False)
            img_fig.axes.get_yaxis().set_visible(False)
            img_to_predict = np.array([X[i]])
            predicted_img = np.argmax(model.predict(img_to_predict), axis=1)
            plt.title(str(class_names[predicted_img[0]]))
            img_fig.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))

        plt.savefig('predictions.png')
        plt.show()





IMG_DIR = "C:/Users/steph/Desktop/train/"
MODEL_NAME = 'cats_and_dogs'

get_color_img = True
create_data = False
save_data = True
IMG_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 0.001

if create_data:
    X, y, class_names = get_X_y(IMG_DIR, get_color_img, IMG_SIZE)
    if save_data:
        save_X_y_class_names(X, y, class_names, MODEL_NAME)
else:
    X, y, class_names = load_X_y_class_names(MODEL_NAME)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)



#ADD your model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3,3), activation='relu', padding='same', input_shape = X_train.shape[1:]))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(BatchNormalization())

model.add(Conv2D(64, kernel_size=(3,3), activation='relu', padding='same'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(BatchNormalization())

model.add(Conv2D(128, kernel_size=(3,3), activation='relu', padding='same'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(BatchNormalization())

model.add(Conv2D(256, kernel_size=(3,3), activation='relu', padding='same'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(BatchNormalization())

model.add(Conv2D(len(class_names), kernel_size=(3,3), activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(BatchNormalization())

model.add(GlobalAveragePooling2D())
model.add(Activation("softmax"))

opt = Adam(learning_rate=LEARNING_RATE)
model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

time1 = time.time()
model.fit(X_train,
          y_train,
          validation_data=(X_test,y_test),
          epochs=EPOCHS,
          batch_size = 4)
print("Time elapsed: ", round(time.time()-time1, 2), "s")

model.save(MODEL_NAME + "_" + str(int(time.time())))

evaluation = evaluate_model(model, X_test, y_test, print_eval=True)
print(evaluation)

model_show_predicted_images(model, X_test, y_test, class_names)
