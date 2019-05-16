from __future__ import unicode_literals

import argparse
import json
import utils

from keras.callbacks import ModelCheckpoint
from keras.layers import Activation, Dense, Dropout, Flatten, Lambda, Conv2D, BatchNormalization
from keras.layers.convolutional import MaxPooling2D
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from utils import *
from math import ceil


kSEED = 10
SIDE_STEERING_CONSTANT = 0.3
NUM_BINS = 23

project_path = '/home/workspace/CarND-Behavioral-Cloning-P3/'
data_path = project_path + 'data_collection/'
BATCH_SIZE = 128
EPOCH_NUM = 13

def generate_batches(images, angles, process_data=True, batch_size=64):
    """
    Keras Batch Generator to create a generator of training examples for model.
    :param images: Training image data.
    :param angles: Angle data for images.
    :param batch_size: Batch size of each training run.
    :param process_data: If the data should be augmented.
    :return: A batch generator.
    """
    batch_images = []
    batch_angles = []
    sample_count = 0

    num_samples = len(angles)
    batch_samples = []
    while True:

        shuffle(images)
        for offset in range(0, num_samples, batch_size):
            batch_samples = images.iloc[offset:offset+batch_size]
            i = 0
            for i in range (0, batch_size):
                center_path = images.iloc[offset + i]['Center'].replace(' ', '')   
                left_path = images.iloc[offset + i]['Left'].replace(' ', '')
                right_path = images.iloc[offset + i]['Right'].replace(' ', '')
                angle = float(angles.iloc[offset + i])
                i = i + 1

                center_image = utils.load_image(center_path)
                batch_images.append(center_image)
                batch_angles.append(angle)

                if process_data:
                    # Image (2) -> Flip the image and invert angle respectively.
                    flipped_image = utils.flip_image(center_path)
                    flipped_angle = -1.0 * angle
                    batch_images.append(flipped_image)
                    batch_angles.append(flipped_angle)

                    # Image (3) -> Tint the center image to random brightness.
                    tint_image = utils.tint_image(center_path)
                    tint_angle = angle
                    batch_images.append(tint_image)
                    batch_angles.append(tint_angle)

                    # Image (4) -> Jitter the center image to make it seem like
                    # different position on the road.
                    jittered_image, jitter_angle = utils.jitter_image(center_path, angle)
                    batch_images.append(jittered_image)
                    batch_angles.append(jitter_angle)

                    # Image (5) -> Load the left image and add steering constant to
                    # compensate.
                    left_image = utils.load_image(left_path)
                    left_angle = min(1.0, angle + SIDE_STEERING_CONSTANT)
                    batch_images.append(left_image)
                    batch_angles.append(left_angle)

                    # Image (6) -> Load the right image and subtract steering
                    # constant to compensate.
                    right_image = utils.load_image(right_path)
                    right_angle = max(-1.0, angle - SIDE_STEERING_CONSTANT)
                    batch_images.append(right_image)
                    batch_angles.append(right_angle)

                yield np.array(batch_images), np.array(batch_angles)


def create_nvidia_model(lr=1e-3, activation='relu', epoch_count=10):

    model = Sequential()
    # Lambda layer normalizes pixel values between 0 and 1
    model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(80, 160, 3)))
    # Convolutional layer (1)
    model.add(Conv2D(24, (5,5), padding='same', activation=activation, strides=(2,2)))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(1, 1)))
    model.add(BatchNormalization())
    # Convolutional layer (2)
    model.add(Conv2D(36, (5,5), padding='same', activation=activation, strides=(2,2)))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(1, 1)))
    model.add(BatchNormalization())
    # Convolutional layer (3)
    model.add(Conv2D(48, (5,5), padding='same', activation=activation, strides=(2,2)))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(1, 1)))
    model.add(BatchNormalization())
    # Convolutional layer (4)
    model.add(Conv2D(64, (3,3), padding='same', activation=activation, strides=(1,1)))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(1, 1)))
    model.add(BatchNormalization())
    # Convolutional layer (5)
    model.add(Conv2D(64, (3,3), padding='same', activation=activation, strides=(1,1)))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(1, 1)))
    model.add(BatchNormalization())
    # Flatten Layer
    model.add(Flatten())
    # Dense Layer (1)
    model.add(Dense(1164, activation=activation))
    # Dense layer (2)
    model.add(Dense(100, activation=activation))
    # Dense layer (3)
    model.add(Dense(50, activation=activation))
    # Dense layer (4)
    model.add(Dense(10, activation=activation))
    # Dense layer (5)
    model.add(Dense(1))
    # Compile model
    model.compile(optimizer=Adam(lr=lr, decay=lr / epoch_count), loss='mse')
    return model


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--lr", help="Initial learning rate",
                           type=float, default=1e-3, required=False)
    argparser.add_argument("--epoch_count", help="Number of epochs to train for",
                           type=int, default= EPOCH_NUM, required=False)
    argparser.add_argument("--activation", help="Activation function to use",
                           type=str, default='relu', required=False)
    args = argparser.parse_args()

    file_name = data_path + 'driving_log.csv'
    columns = ['Center','Left','Right','Steering','Throttle','Break','Speed']

    print('[INFO] Loading Data.')
    images, angles = utils.load_data(file_name, columns)

    print('[INFO] Creating Training and Testing Data.')
    X_train, X_val, y_train, y_val = train_test_split(images, angles, test_size=0.15, random_state=kSEED)

    print('[INFO] Preprocessing Images and Data Augmentation.')
    train_batch = generate_batches(X_train, y_train, process_data=True, batch_size = BATCH_SIZE)
    valid_batch = generate_batches(X_val, y_val, process_data=False, batch_size = BATCH_SIZE)

    print('[INFO] Creating Model.')
    model = create_nvidia_model(args.lr, args.activation, args.epoch_count)

    print('[INFO] Training Model.')
    model.fit_generator(
        train_batch,
        steps_per_epoch = ceil(len(y_train)/BATCH_SIZE),
        epochs=args.epoch_count,
        validation_data=valid_batch,
        validation_steps= ceil(len(y_val)/BATCH_SIZE),
        verbose=1)

    print('[INFO] Saving Model')
    model.save('model.h5')
    with open('model.json', 'w') as outfile:
        json.dump(model.to_json(), outfile)