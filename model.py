import itertools
import tensorflow as tf
import numpy as np
from keras.layers import Input, Flatten, Dense
from keras.models import Model
from scipy import misc
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, MaxPooling2D, Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.advanced_activations import ELU
from keras.optimizers import Adam
import cv2
import math
import json
import matplotlib.pyplot as plt


from sklearn.utils import shuffle


def load_data():
    # Get center, left, and right pics
    center_pics = np.genfromtxt('driving_log.csv', delimiter=',', usecols=(0,), unpack=True, dtype=str,skip_header = 1) 
    left_pics = np.genfromtxt('driving_log.csv', delimiter=',', usecols=(1,), unpack=True, dtype=str, skip_header=1)
    right_pics = np.genfromtxt('driving_log.csv', delimiter=',', usecols=(2,), unpack=True, dtype=str, skip_header=1)
    steering_angle = np.genfromtxt('driving_log.csv', delimiter=',', usecols=(3,), unpack=True, dtype=str,skip_header=1)

    X_center = center_pics
    X_left = left_pics
    X_right = right_pics
    y_train = steering_angle
                
    X_center, X_left, X_right, y_train = shuffle(X_center, X_left, X_right, y_train)
    y_train = y_train.astype(np.float)
                        
    center_train, center_val, left_train, left_val, right_train, right_val, y_train, y_val = train_test_split(X_center,X_left,X_right,y_train,test_size=0.10,random_state=832289)
    
    return center_train, center_val, left_train, left_val, right_train, right_val, y_train, y_val

def get_model():
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1.,input_shape=(64,64,3),output_shape=(64,64,3)))
                     
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
        
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
                
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
        
    model.add(Convolution2D(64, 3, 3, border_mode="same"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
            
    model.add(Convolution2D(64, 3, 3, border_mode="same"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
            
    model.add(Flatten())
    model.add(Dropout(.2))
    model.add(Activation('relu'))
    model.add(Dense(1164))
    model.add(Dropout(.5))
    model.add(Activation('relu'))
    model.add(Dense(100))
    model.add(Dropout(.5))
    model.add(Activation('relu'))
    model.add(Dense(50))
    model.add(Dropout(.5))
    model.add(Activation('relu'))
    model.add(Dense(10))
    model.add(Dropout(.5))
    model.add(Activation('relu'))
    model.add(Dense(1))

    model.summary()
    adam_op = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='mean_squared_error', optimizer=adam_op, metrics=['mean_absolute_error'])

    return model

def crop_resize(image):
    #Resize image to 64x64 and crop top and bottom 20% to make interpretation easier for model
    shape = image.shape
    image = image[int(shape[0] * 0.2):int(shape[0] * 0.8), 0:shape[1]]
    image = cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA)
    
    return image

def random_brightness(image):
    # Apply random brightness to image
    bright_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    bright_image = cv2.cvtColor(bright_image,cv2.COLOR_RGB2HSV)
    bright_image[:,:,2] = bright_image[:,:,2]*(.25+np.random.uniform())
    bright_image= cv2.cvtColor(bright_image,cv2.COLOR_HSV2RGB)
    return bright_image

def get_generator(train_features, left_features, right_features, train_labels, batch_size):
    # Yield batch at a time to fit_generator using random data. Data will be chosen randomly between center, left, and right
    # images, and steering angle and images will be augmented
    images = np.zeros((batch_size, 64, 64, 3))
    angles = np.zeros(batch_size)
    while 1:
        for i in range(batch_size):
            # Generate random index value from batch
            rand_line = np.random.randint(len(train_labels))
            # Get a batch of training features and labels.
            image, label = get_rand(train_features[rand_line], left_features[rand_line], right_features[rand_line], train_labels[rand_line])
            if label != 0:
                # Random flip image if it is not at 0 angle
                image, label = flip(image, label)

            images[i] = image
            angles[i] = label
        yield images, angles

def get_rand(center_pic, left_pic, right_pic, angle):
    # Randomly chooses between center, left, or right picture and adjust steering angle
    
    rand_pic = np.random.randint(3)
    if (rand_pic == 0):
        path = left_pic.strip()
        angle += 0.3
    if (rand_pic == 1):
        path = center_pic.strip()
    if (rand_pic == 2):
        path = right_pic.strip()
        angle -= 0.275
    # Apply brightness and image shifting augmentation techniques
    image = cv2.imread(path)
    image = random_brightness(image)
    image = crop_resize(image)
    return image, angle


def flip(image, label):
    # Randomly flips image and negates steering angle
    ind_flip = np.random.randint(2)
    if ind_flip == 0:
        image = cv2.flip(image, 1)
        label = - label
    return image, label

def main():
    center_train, center_val, left_train, left_val, right_train, right_val, y_train, y_val = load_data()
    
    model = get_model()
    
    threshold = 1
    batch_size = 512
    
    for i in range(14):
        # Obtain training and validation generators and use fit_generator to test data. Incrementally reduce threshold
        # to eliminate extreme values
        train_r_generator = get_generator(center_train, left_train, right_train, y_train, batch_size)
        val_r_generator = get_generator(center_val, left_val, right_val, y_val, batch_size)
        model.fit_generator(train_r_generator, samples_per_epoch=25600, nb_epoch=1, validation_data=val_r_generator, nb_val_samples=10, verbose=1)

        print("Finished Epoch #", (i + 1))
    
    json_string = model.to_json()
    with open('model.json', 'w') as outfile:
        json.dump(json_string, outfile)
    
    model.save("model.h5")
    
    pass


if __name__ == '__main__':
    main()
