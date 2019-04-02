import pandas as pd
import numpy as np
import os
import pickle
from sklearn.cluster import KMeans
## data pariolo

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

def fitandlabel(clus_num, data_train_flat, response_train):

    'fits certian number of clusters to a data frame and returns the df with'
    'new features for labels, and surface (based on their id connections with response_train)'
    'and it also returns de freqs of the categories by cluster'

    'Asegurarse de que data_train_flat solo tiene int64 en las columnas'
    'tambien comprobar que .index del dataframe corresponse con series_id'

    ##lets fit the data to 9 clusters
    kmeans = KMeans(n_clusters=clus_num, random_state=0)

    #let's check how we can map the fitted clusters
    km_fit = kmeans.fit(data_train_flat.drop('series_id', axis=1))
    clusters_train = pd.DataFrame(km_fit.labels_)
    data_train_flat['labels'] = clusters_train

    #let's introduce the surface variable from response_train in our flatted dataset
    data_train_flat.index = data_train_flat['series_id']
    data_train_flat['surface'] = response_train['surface']

    #let's get the most frequent categories by each cluster into a dictionary, so we can map it later
    freqcat_train = data_train_flat.groupby('labels')['surface'].value_counts()
    return data_train_flat, freqcat_train, km_fit


def freqcat_todict(x):
    dict_this_clus = {}

    for j in range(9):
        index_max = x[j] == x[j].max()
        name_surface = x[j].index[index_max]
        dict_this_clus[j] = name_surface[0]

    return dict_this_clus


data_train_flat_fit, freqcat_train, km_fit = fitandlabel(9, data_train_flat, response_train)
mapping = freqcat_todict(freqcat_train)


data_test_flat.index = data_test_flat['series_id']
def predandmap(data_test_flat, km_fit):
    km_pred = km_fit.predict(data_test_flat.drop('series_id', axis=1))
    y_pred = pd.DataFrame(km_pred)
    data_test_flat['labels'] = y_pred
    data_test_flat['surface'] = data_test_flat['labels'].map(mapping)

    return data_test_flat[['series_id', 'surface']]


submission = predandmap(data_test_flat, km_fit)
submission.to_csv('/home/samuel/Work/coffe2/data/datacc2019/submissions/submission9.csv')