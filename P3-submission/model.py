import os
import cv2
import numpy as np
import csv
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random

from keras.models import Sequential, Model
from keras.layers import Input, Convolution2D, Flatten, Dense, Dropout, Activation, Lambda, Cropping2D

CONFIG = {
    'batchsize': 32,
    'input_width': 320,
    'input_height': 160,
    'input_channels': 3,
    'correction': 0.15,
    #'correction': 0.2,
    'cropping': ((50,20), (0,0))
}

def load_and_split_data(data_path, test_size=0.2):
    with open(data_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        samples = [line for line in reader][1:]
    train_samples, validation_samples = train_test_split(samples, test_size=test_size, random_state=0)
    return train_samples, validation_samples

def generator(samples, img_path, side_camera=False, augment_data=False, batch_size=32):
    num_samples = len(samples)
    while 1:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                center, left, right, steering, throttle, brake, speed = batch_sample
                
                steering_center = float(steering)

                if side_camera:
                    r = random.choice([0,1,2])
                    if r ==0:
                        angle = steering_center + CONFIG['correction']
                        image = cv2.cvtColor(cv2.imread(os.path.join(img_path, left.strip())), cv2.COLOR_BGR2RGB)
                    elif r==2:
                        angle = steering_center
                        image = cv2.cvtColor(cv2.imread(os.path.join(img_path, center.strip())), cv2.COLOR_BGR2RGB)
                    else:
                        angle = steering_center - CONFIG['correction']
                        image = cv2.cvtColor(cv2.imread(os.path.join(img_path, right.strip())), cv2.COLOR_BGR2RGB)
                else:
                    angle = steering_center
                    image = cv2.cvtColor(cv2.imread(os.path.join(img_path, center.strip())), cv2.COLOR_BGR2RGB)

                images.append(image)
                angles.append(angle)

                if augment_data:
                    images.append(cv2.flip(image, 1))
                    angles.append(angle*-1.0)


            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)
            
def nvidia_model(summary=True):
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(CONFIG['input_height'], CONFIG['input_width'], CONFIG['input_channels'])))
    model.add(Cropping2D(cropping=CONFIG['cropping']))
    model.add(Convolution2D(24,5,5, subsample=(2,2),activation='relu'))
    model.add(Convolution2D(36,5,5, subsample=(2,2),activation='relu'))
    model.add(Convolution2D(48,5,5, subsample=(2,2),activation='relu'))
    model.add(Convolution2D(64,3,3, activation='relu'))
    model.add(Convolution2D(64,3,3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dropout(0.8))
    model.add(Dense(50))
    #model.add(Dropout(0.8))
    model.add(Dense(10))
    model.add(Dense(1))

    if summary:
        model.summary()
    
    return model

if __name__ == "__main__":
    train_samples, validation_samples = load_and_split_data(data_path='./data/data/driving_log.csv', test_size=0.2)

    train_generator = generator(train_samples, img_path='./data/data', side_camera=True, augment_data=True,  batch_size=CONFIG['batchsize'])
    validation_generator = generator(validation_samples, img_path='./data/data', batch_size=CONFIG['batchsize'])

    model = nvidia_model(summary=False)
    model.compile(optimizer='adam', loss='mae')
    #model.compile(optimizer-'adam', loss='mse')

    model.fit_generator(train_generator, samples_per_epoch=len(train_samples), 
                                        validation_data=validation_generator, nb_val_samples=len(validation_samples), 
                                        nb_epoch=10, verbose=1) #nb_epoch=20

    model.save('model.h5')
