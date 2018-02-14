import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier


df_train = pd.read_csv('raw_data/training_data.csv')

reindexed_list = ['id', 'game', 'white', 'black', 'white_elo', 'black_elo', 'white_rd',
                'black_rd', 'whiteiscomp', 'blackiscomp', 'timecontrol', 'date', 'time',
                'white_clock', 'black_clock', 'eco', 'plycount', 'moves',
                'commentaries']
df_train = df_train.reindex(reindexed_list, axis=1)

# Split into testing and training set
X = df_train.iloc[:, 0:18]
y = df_train['commentaries']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10)

# Functin to process_data into appropriate form.......
def process_data(X_dummy):
    # features to drop instantly
    features_to_drop = ['id', 'game', 'white', 'black', 'date', 'black_clock']
    X_dummy.drop(features_to_drop, axis=1, inplace=True)

    # map whiteiscomp and blackiscomp to binary
    X_dummy['whiteiscomp'] = X_dummy['whiteiscomp'].map({False: 0,
                                                     True:1})
    X_dummy['blackiscomp'] = X_dummy['blackiscomp'].map({False: 0,
                                                     True:1})

    # create new feature elo_difference
    X_dummy['elo_dif'] = abs(X_dummy['white_elo'] - X_dummy['black_elo'])

    # fix the time control feature
    time_control = X_dummy['timecontrol']
    fixed_time = []
    extended_time = []
    for time in time_control:
        time_tuple = time.split('+')
        fixed_time.append(int(time_tuple[0]))
        extended_time.append(int(time_tuple[1]))
    X_dummy['fixed_time'] = pd.Series(fixed_time, index=X_dummy.index)
    X_dummy['extended_time'] = pd.Series(extended_time, index=X_dummy.index)

    # some more features to drop
    X_dummy.drop(['timecontrol', 'moves', 'eco', 'white_clock'], axis=1, inplace=True)

    # fix the time feature
    time_period = []
    for period in X_dummy['time']:
        time_list = period.split(':')
        time_period.append((100*int(time_list[0])) + (int(time_list[1])) + (int(time_list[2])/100))
    X_dummy['total_time'] = pd.Series(time_period, index=X_dummy.index)

    # Drop the time feature
    X_dummy.drop(['time'], axis=1, inplace=True)
    return X_dummy

# map y values before fitting into model
def map_y_values(y):
    y = y.map({'Black checkmated':-2, 'White resigns':3, 'White checkmated':2,
       'Black resigns':-3, 'White forfeits on time':4,
       'Black forfeits by disconnection':-5,
       'Neither player has mating material':-7, 'Black forfeits on time':-4,
       'Game drawn by repetition':1, 'Game drawn by the 50 move rule':0,
       'Game drawn by mutual agreement':-1,
       'White forfeits by disconnection':5,
       'Black ran out of time and White has no material to mate':6,
       'White ran out of time and Black has no material to mate':-6,
       'Game drawn by stalemate':7})
    return y


# Make changes to model here
def fit_model(X, y):
    clf = RandomForestClassifier()
    clf.fit(X,y)
    return clf


# Invert the mapping before submission
def invert_mapping(y):
    forward_map = {'Black checkmated':-2, 'White resigns':3, 'White checkmated':2,
       'Black resigns':-3, 'White forfeits on time':4,
       'Black forfeits by disconnection':-5,
       'Neither player has mating material':-7, 'Black forfeits on time':-4,
       'Game drawn by repetition':1, 'Game drawn by the 50 move rule':0,
       'Game drawn by mutual agreement':-1,
       'White forfeits by disconnection':5,
       'Black ran out of time and White has no material to mate':6,
       'White ran out of time and Black has no material to mate':-6,
       'Game drawn by stalemate':7}
    inv_map = {v:k for k,v in forward_map.items()}
    y = y.map(inv_map)
    return y
