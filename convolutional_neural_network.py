import pandas as pd
import numpy as np
import os
import pickle
import numpy as np
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.layers import Reshape
from keras.layers import Reshape




from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from kt_utils import *

import keras.backend as K
K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

## data

cwd = os.chdir('/home/samuel/Work/coffe2/data/datacc2019')

data_train_path = '/home/samuel/Work/coffe2/data/datacc2019/X_train.csv'
response_train_path = '/home/samuel/Work/coffe2/data/datacc2019/y_train.csv'
test_data_path = '/home/samuel/Work/coffe2/data/datacc2019/X_test.csv'
sub_data_path = '/home/samuel/Work/coffe2/data/datacc2019/sample_submission.csv'

data_train = pd.read_csv(data_train_path)
response_train = pd.read_csv(response_train_path)
data_test = pd.read_csv(test_data_path)
sub_data = pd.read_csv(sub_data_path)

#flat sets can be substituted by different sets, we could try grouping by means again, and adding std, etc

data_test_flat = pd.read_pickle('/home/samuel/Work/coffe2/data/datacc2019/data_test_flat.pkl')
data_train_flat = pd.read_pickle('/home/samuel/Work/coffe2/data/datacc2019/data_train_flat.pkl')

data_train_flat.index = data_train_flat['series_id']


# We are going to try to fit our data to a convolutional neural network and see how it perform

#lets first try to properly process the data train so it fits the function

# Lo ideal, seria que las dimensiones del input fueran (128,3,3) = (n_x, n_y, n_c) , siendo n_c
# el numero de canales, siendo cada canal correspondiente a la orientacion, la acceleracion y la velocidad
#para la orientacion, como son cuatro columnas, habria que pasar las coordenadas (quaterniones) a "euler angles" (esta explicat en kaggle)
#para que sean tres columnas

data_train_flat_toarray = data_train_flat.drop("series_id", axis = 1)
input_train = np.zeros((3810,1, 128, 10))

for i in range(data_train_flat_toarray.shape[0]):
    current_row = np.array(data_train_flat_toarray.iloc[i:(i+1), ])
    print(i)
    current_row =  current_row.reshape(128, 10)
    input_train[i,...] = current_row

def cnn_proba(input_shape):
    """
    Implementation of the model.

    Arguments:
    input_shape -- shape of the images of the dataset

    Returns:
    model -- a Model() instance in Keras
    """

    ### START CODE HERE ###
    X_input = Input(input_shape)

    # Zero-Padding: pads the border of X_input with zeroes
    X = ZeroPadding2D((3, 3))(X_input)

    # CONV -> BN -> RELU Block applied to X
    X = Conv2D(32, (5, 5), strides=(1, 1), name='conv0')(X)
    X = BatchNormalization(axis=3, name='bn0')(X)
    X = Activation('relu')(X)

    # MAXPOOL
    X = MaxPooling2D((2, 2), name='max_pool')(X)

    # FLATTEN X (means convert it to a vector) + FULLYCONNECTED
    X = Flatten()(X)
    X = Dense(1, activation='sigmoid', name='fc')(X)

    # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
    model = Model(inputs=X_input, outputs=X, name='cnn_proba')

    ### END CODE HERE ###

    return model


#we first create a dummy variable for the response Y
dummies = pd.get_dummies(response_train.surface)


cnn_model = cnn_proba((128,10))
cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
cnn_model.fit(input_train, dummies, epochs = 35, batch_size = 16)

#hay que solucionar: ValueError: Error when checking input: expected input_2 to have shape (128, 10, 1) but got array with shape (1, 128, 10)
#natros tenim les dimensions de input com a (3810, 1, 128, 10), pero tindrien que ser (3810, 128, 10, 1)
