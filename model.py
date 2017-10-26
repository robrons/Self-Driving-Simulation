import os
import pandas as pd
import numpy as np
import argparse

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.layers import Lambda, Dropout, Conv2D, MaxPool2D, Dense, Flatten
from utils import INPUT_SHAPE, batch_generator


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

def train_model(model, dataArgs, X_train, X_test, Y_train, Y_test):

    #Saves onlu the best model in .h5 format

    checkpoint = ModelCheckpoint('model-{epoch:03d}.h5',
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=dataArgs.save_best_only,
                                 mode='auto')

    #We trian the model using mean squared error as our loss function and Adams which is Gradient decent as out optimizer.

    model.compile(loss='mean_squared_error', optimizer=Adam(lr=dataArgs.learning_rate))

    #Helps parrallize data augumentation and model trianing
    model.fit_generator(batch_generator(dataArgs.data_dir, X_train, Y_test, dataArgs.batch_size, True),
                        dataArgs.samples_per_epoch,
                        dataArgs.nb_epoch,
                        max_q_size=1,
                        validation_data=batch_generator(dataArgs.data_dir, X_test, Y_test, dataArgs.batch_size, False),
                        nb_val_samples=len(X_test),
                        callbacks=[checkpoint],
                        verbose=1)

#for command line args
def s2b(s):
    """
    Converts a string to boolean value
    """
    s = s.lower()
    return s == 'true' or s == 'yes' or s == 'y' or s == '1'


def main():
    """
    Load train/validation data set and train the model
    """
    parser = argparse.ArgumentParser(description='Behavioral Cloning Training Program')
    parser.add_argument('-d', help='data directory',        dest='data_dir',          type=str,   default='data')
    parser.add_argument('-t', help='test size fraction',    dest='test_size',         type=float, default=0.2)
    parser.add_argument('-k', help='drop out probability',  dest='keep_prob',         type=float, default=0.5)
    parser.add_argument('-n', help='number of epochs',      dest='nb_epoch',          type=int,   default=10)
    parser.add_argument('-s', help='samples per epoch',     dest='samples_per_epoch', type=int,   default=20000)
    parser.add_argument('-b', help='batch size',            dest='batch_size',        type=int,   default=40)
    parser.add_argument('-o', help='save best models only', dest='save_best_only',    type=s2b,   default='true')
    parser.add_argument('-l', help='learning rate',         dest='learning_rate',     type=float, default=1.0e-4)
    args = parser.parse_args()

    #print parameters
    print('-' * 30)
    print('Parameters')
    print('-' * 30)
    for key, value in vars(args).items():
        print('{:<20} := {}'.format(key, value))
    print('-' * 30)

    #load data
    data = loadData(args)
    #build model
    model = buidModel(args)
    #train model on data, it saves as model.h5 
    train_model(model, args, *data)


if __name__ == '__main__':
    main()
