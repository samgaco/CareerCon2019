import pandas as pd
import numpy as np
import os
import pickle

## data

Sam_Dir = '/home/samuel/Work/coffe2/data/datacc2019'
Ces_Dir = 'D:/Estudi/Uni/Actual/z_Altres/Python/projects/career_con_2019/Data'
# Elegir directori segons usuari
Name_Dir = Ces_Dir

def load_data(dir_name):
    cwd = os.chdir(dir_name)
    
    data_train = pd.read_csv(dir_name + '/X_train.csv')
    response_train = pd.read_csv(dir_name + '/y_train.csv')
    data_test = pd.read_csv(dir_name + '/X_test.csv')
    sub_data = pd.read_csv(dir_name + '/sample_submission.csv')
    
    return data_train, response_train, data_test, sub_data

data_train, response_train, data_test, sub_data = load_data(Name_Dir)


### Let's use this script to make some feature engineering
### We are going to create a new table, each row will consist on the whole set of measurements for a same series_id
### We then will cluster as has been done in the data_analysis.ipynb


#First let's create a function that will select the 'series_id', and flat all the vectors of measurements.

#We keep the ids in a new variable, this transformation will have to be made over the test as well
data_train_ids = data_train[['row_id', 'series_id', 'measurement_number']]
data_test_ids = data_train[['row_id', 'series_id', 'measurement_number']]

#we drop the unecessary ids, and keep the series_id for the selection later
data_train_dropsid = data_train.drop(['row_id', 'measurement_number'], axis=1)
data_test_dropsid = data_train.drop(['row_id', 'measurement_number'], axis=1)


def vector_meas(x, name_id):

    'This function takes all instances in certain id and flats them all converting them into '
    'A new data set' \
    'Takes two values: x = the dataset , name_id  = the name of the id from which to take the instances '


    for i in x[name_id].unique():

        x_select = x[x[name_id] == i]
        x_select = x_select.drop([name_id],axis=1)
        x_select = x_select.values.flatten()
        x_select = pd.DataFrame(x_select).transpose()

        if i == 0:
            data_train_flat = x_select
        else:
            data_train_flat = pd.concat([data_train_flat, x_select], axis = 0)

    return data_train_flat

#we create this new data sets and assign them ids
data_train_flat = vector_meas(data_train_dropsid, 'series_id')
data_train_flat['series_id'] = range(data_train_flat.shape[0])

data_test_flat = vector_meas(data_test_dropsid, 'series_id')
data_test_flat['series_id'] = range(data_test_flat.shape[0])


#let's save those new data_sets in pickle formats
data_train_flat.to_pickle(Name_Dir + '/data_train_flat.pkl')
data_test_flat.to_pickle(Name_Dir + '/data_test_flat.pkl')