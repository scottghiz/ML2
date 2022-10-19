#!/usr/bin/env python
# coding: utf-8

from datetime import datetime
import boto3
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sagemaker
from sagemaker import get_execution_role
from sagemaker.inputs import TrainingInput
from sagemaker.serializers import CSVSerializer

from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn import preprocessing
le = preprocessing.LabelEncoder()

import pandas as pd
import numpy as np
import datetime
import re
import time
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', 200)
pd.set_option('display.max_rows', 500)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import xgboost as xgb
from math import sqrt
from sklearn.metrics import mean_squared_error
from scipy.stats import uniform, randint
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
import pydot_ng as pydot
from IPython.display import Image
import math
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Activation, Reshape
from keras.layers import Concatenate, Dropout

from matplotlib import pyplot

from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from sklearn.model_selection import GridSearchCV

import pickle
from joblib import dump
from joblib import load

from sklearn.preprocessing import OneHotEncoder as SklearnOneHotEncoderA

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.datasets import make_classification
#from eds_encoders import OneHotEncoder as edsohe
#from mlResult import MLResult

#from keras.layers.embeddings import Embedding
#from keras.utils import plot_model
#from keras.optimizers import SGD

##############################################################

# load data
#dataset = loadtxt('../trino_ml.csv', delimiter=",")

df_dataset = pd.read_csv('/home/scripts/PREDICT/TRAINING_DATA/CSV/trino_ml.csv')

### REMOVE SPECIFIC COLUMNS (DUPLICATES, GREEN LIGHT/GREEN LIGHT, ETC.) ###
del df_dataset["Unnamed: 0"]
del df_dataset["accountnumber"]
del df_dataset["day_id"]
del df_dataset["ticket_id"]
del df_dataset["doc_id"]

del df_dataset["itg_title"]
del df_dataset["proactiveSwapSubtype"]
del df_dataset["docsisupsfail"]
del df_dataset["docsisdnsfail"]
del df_dataset["spectraresult"]

del df_dataset["wo_ps_model"]
del df_dataset["jobTypeDescription"]
del df_dataset["epon"]
del df_dataset["fmfaildesc"]
del df_dataset["isDropAndGoJob"]
del df_dataset["isX1Account"]
del df_dataset["isAllWirelessIPVideo"]
del df_dataset["isXfinityFlexAccount"]
del df_dataset["isMultiTenantGatewayAccount"]


#encode_columns = ['wo_ps_model','finalintentcode','resulttype','warningtype','looseconnaccsts','epon','docsisdnsfail','docsisupsfail','lcfail','spectrafail','fluxfail','fmfail','fmfaildesc','mocafail','ntwseg','bridgentw','alloutfail','spectraresult','itg_title','isWorkorderCustomerFacing','isDropAndGoJob','jobTypeDescription','hasGram','isX1Account','isAllWirelessIPVideo','isXfinityFlexAccount','isMultiTenantGatewayAccount','proactiveSwapSubtype','proactiveSwapMsg']
encode_columns = ['finalintentcode','resulttype','warningtype','looseconnaccsts','lcfail','spectrafail','fluxfail','fmfail','mocafail','ntwseg','bridgentw','alloutfail','isWorkorderCustomerFacing','hasGram','proactiveSwapMsg']
encode_df = df_dataset[encode_columns]

encode_df = encode_df.astype('str')
encode_df = encode_df.apply(le.fit_transform)
score_encode_drop = df_dataset.drop(encode_columns, axis = 1)
score_encode = pd.concat([score_encode_drop, encode_df], axis = 1)

df_encode = score_encode
df_encode.to_csv("/home/scripts/PREDICT/ML_TRAINING/CSV/trino_encoded.csv")

#df_encode = df_encode[['wo_ps_model','finalintentcode','resulttype','warningtype','actaccdev','devtested','faileddev','devcount','looseconnaccsts','epon','docsisdnsfail','docsisupsfail','lcfail','spectrafail','fluxfail','fmfail','fmfaildesc','mocafail','ntwseg','bridgentw','mocadevices','alloutfail','spectraresult','itg_title','isWorkorderCustomerFacing','isDropAndGoJob','jobTypeDescription','hasGram','isX1Account','isAllWirelessIPVideo','isXfinityFlexAccount','isMultiTenantGatewayAccount','proactiveSwapSubtype','proactiveSwapMsg','truck_roll_scheduled']]
df_encode = df_encode[['finalintentcode','resulttype','warningtype','actaccdev','devtested','faileddev','devcount','looseconnaccsts','lcfail','spectrafail','fluxfail','fmfail','mocafail','ntwseg','bridgentw','mocadevices','alloutfail','isWorkorderCustomerFacing','hasGram','proactiveSwapMsg','truck_roll_scheduled']]
#print(df_encode)

df_y = df_encode['truck_roll_scheduled']
df_X = df_encode.drop(['truck_roll_scheduled'], axis=1)

y = df_y.to_numpy()
X = df_X.to_numpy()

# split data into train and test sets
seed = 7
test_size = 0.25
# create example data
X, y = make_classification(n_samples=1000000, 
                           n_informative=5,
                           n_classes=2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

### 

### We need to prepare data as DMatrix objects ###################
train = xgb.DMatrix(X_train, y_train)
test = xgb.DMatrix(X_test, y_test)

# We need to define parameters as dict
params = {
    "learning_rate": 0.01,
    "max_depth": 6
}
# training, we set the early stopping rounds parameter
model_xgb = xgb.train(params, 
          train, evals=[(train, "train"), (test, "validation")], 
          num_boost_round=20000, early_stopping_rounds=20)

print(model_xgb.best_ntree_limit)

print(model_xgb.predict(test))

# save to JSON
model_xgb.save_model("/home/scripts/PREDICT/ML_TRAINING/TRAINED_MODELS/xgb_trained.json")
# save to text format
model_xgb.save_model("/home/scripts/PREDICT/ML_TRAINING/TRAINED_MODELS/xgb_trained.txt")

######################################
import sys ###########################
sys.exit() ### EXIT ##################
######################################

##############################################################
