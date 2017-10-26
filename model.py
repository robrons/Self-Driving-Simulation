import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.layers import Lambda, Dropout, Conv2D, MaxPool2D, Dense, Flatten
from utils import INPUT_SHAPE

def loadData(dataArgs):

    #loading data from csv files

    data = pd.read_csv(os.path.join(dataArgs.path, 'driving_log.csv'))

    #Our input data
    X = data[['center', 'left', 'right']].values

    print(np.shape(X))
    #Our output data
    y = data['steering'].values

    #Splitting the data for treaining and testing (80/20)
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=dataArgs.test_size, random_state=0)

    return X_train, X_test, Y_train, Y_test


def buidModel(dataArgs):

    #using Keras as our framework
    #Model built accoording to the Nvidia Paper Linked in the Project Git

    model = Sequential()

    #image normalization layer

    model.add(Lambda(lambda x: x/127.5-1.0, input_shape=INPUT_SHAPE))

    #added 5 convolutional neural networks

    model.add(Conv2D(24, 5, 5, activation='elu', subsample=(2,2)))
    model.add(Conv2D(36, 5, 5, activation='elu', subsample=(2,2)))
    model.add(Conv2D(48, 5, 5, activation='elu', subsample=(2,2)))

    #removed the sub sampliling as the stride size is 1/1 for the rest

    model.add(Conv2D(64, 3, 3, activation='elu'))
    model.add(Conv2D(64, 3, 3, activation='elu'))

    # Added the drop out model to minimizing overfitting

    model.add(Dropout(dataArgs.keep_prob))

    #Falleting the images to add to fully connected layers

    model.add(Flatten())

    #Adding fully connected layers

    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))

    #final model without elu for steering command

    model.add(Dense(1))
    model.summary()

    return model

