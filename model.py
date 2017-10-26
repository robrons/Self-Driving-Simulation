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

    #Our output data
    y = data['steering'].values

    #Splitting the data for treaining and testing (80/20)
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=dataArgs.test_size, random_state=0)

    return X_train, X_test, Y_train, Y_test

