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

#from keras.layers.embeddings import Embedding
#from keras.utils import plot_model
#from keras.optimizers import SGD

##############################################################

# load data
#dataset = loadtxt('../trino_ml.csv', delimiter=",")

df_dataset = pd.read_csv('/home/scripts/TRINO/CSV/trino_ml.csv')

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
df_encode.to_csv("trino_encoded.csv")

#df_encode = df_encode[['wo_ps_model','finalintentcode','resulttype','warningtype','actaccdev','devtested','faileddev','devcount','looseconnaccsts','epon','docsisdnsfail','docsisupsfail','lcfail','spectrafail','fluxfail','fmfail','fmfaildesc','mocafail','ntwseg','bridgentw','mocadevices','alloutfail','spectraresult','itg_title','isWorkorderCustomerFacing','isDropAndGoJob','jobTypeDescription','hasGram','isX1Account','isAllWirelessIPVideo','isXfinityFlexAccount','isMultiTenantGatewayAccount','proactiveSwapSubtype','proactiveSwapMsg','truck_roll_scheduled']]
df_encode = df_encode[['finalintentcode','resulttype','warningtype','actaccdev','devtested','faileddev','devcount','looseconnaccsts','lcfail','spectrafail','fluxfail','fmfail','mocafail','ntwseg','bridgentw','mocadevices','alloutfail','isWorkorderCustomerFacing','hasGram','proactiveSwapMsg','truck_roll_scheduled']]
#print(df_encode)

df_Y = df_encode['truck_roll_scheduled']
df_X = df_encode.drop(['truck_roll_scheduled'], axis=1)

Y = df_Y.to_numpy()
X = df_X.to_numpy()

# split data into train and test sets
seed = 7
test_size = 0.25
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

# fit model on training data - XGBREGRESSOR TRAINING #
# training
model = xgb.XGBRegressor(n_estimators=50000, max_depth=6, learning_rate=0.01)
model.fit(X_train, y_train, 
          eval_set=[(X_train, y_train), (X_test, y_test)], 
          early_stopping_rounds=20)

print(model.predict(X_test))
print(model.predict(X_test, ntree_limit=model.best_ntree_limit))
# save in JSON format
model.save_model("xgb_sklearn_train.json")
# save in text format
model.save_model("xgb_sklearn_train.txt")

print(model)

# load the model:
model2 = xgb.XGBRegressor()
model2.load_model("xgb_sklearn_train.json")

print(model2)

# optimal number of trees:
print(model2.best_ntree_limit)


######################################
import sys ###########################
sys.exit() ### EXIT ##################
######################################




#model = XGBClassifier()
#model.fit(X_train, y_train)

# feature importance
print(model.feature_importances_)
# plot
pyplot.bar(range(len(model.feature_importances_)), model.feature_importances_)
#pyplot.show()
plt.savefig('/home/scripts/TRINO/ML/feature_importance.png')

print(model)

# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]

# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

# Fit model using each importance as a threshold
thresholds = sorted(model.feature_importances_)
for thresh in thresholds:
	# select features using threshold
	selection = SelectFromModel(model, threshold=thresh, prefit=True)
	select_X_train = selection.transform(X_train)
	# train model
	selection_model = XGBClassifier()
	selection_model.fit(select_X_train, y_train)
	# eval model
	select_X_test = selection.transform(X_test)
	y_pred = selection_model.predict(select_X_test)
	predictions = [round(value) for value in y_pred]
	accuracy = accuracy_score(y_test, predictions)
	print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (thresh, select_X_train.shape[1], accuracy*100.0))

# plot feature importance
xgb.plot_importance(model)
plt.savefig('/home/scripts/TRINO/ML/feature_importance_sort.png')

#################################################
### k-Fold Cross Validation ###
# CV model
model = xgb.XGBClassifier()
kfold = KFold(n_splits=10, random_state=7, shuffle=True)
results = cross_val_score(model, X_test, y_test, cv=kfold)
print()
print("Accuracy (train set): %.2f%%" % (accuracy * 100.0))
print("Accuracy (k-Fold cross validation): %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))


### hyperparameter tuning - TEST DATA ###
dmatrix = xgb.DMatrix(data=X_test, label=y_test)
params={ 'objective':'reg:squarederror',
         'max_depth': 6, 
         'colsample_bylevel':0.5,
         'learning_rate':0.01,
         'random_state':20}
#params={'objective':'reg:squarederror'}
#cv_results = xgb.cv(dtrain=dmatrix, params=params, nfold=10, metrics={'rmse'}, as_pandas=True, seed=20)
cv_results = xgb.cv(dtrain=dmatrix, params=params, nfold=10, metrics={'rmse'}, as_pandas=True, seed=20, num_boost_round=1000)
print('RMSE (test): %.2f' % cv_results['test-rmse-mean'].min())

### hyperparameter tuning - TRAINING DATA ###
dmatrix = xgb.DMatrix(data=X_train, label=y_train)
params={ 'objective':'reg:squarederror',
         'max_depth': 6, 
         'colsample_bylevel':0.5,
         'learning_rate':0.01,
         'random_state':20}
#params={'objective':'reg:squarederror'}
#cv_results = xgb.cv(dtrain=dmatrix, params=params, nfold=10, metrics={'rmse'}, as_pandas=True, seed=20)
cv_results = xgb.cv(dtrain=dmatrix, params=params, nfold=10, metrics={'rmse'}, as_pandas=True, seed=20, num_boost_round=1000)
print('RMSE (train): %.2f' % cv_results['train-rmse-mean'].min())

##################################################
#### Grid Search (hyperparameter tuning, test) ###
#params = { 'max_depth': [3,6,10],
#           'learning_rate': [0.01, 0.05, 0.1],
#           'n_estimators': [100, 500, 1000],
#           'colsample_bytree': [0.3, 0.7]}
#xgbr = xgb.XGBRegressor(seed = 20)
#clf = GridSearchCV(estimator=xgbr, 
#                   param_grid=params,
#                   scoring='neg_mean_squared_error', 
#                   verbose=1)
#clf.fit(X_test, y_test)
#print("Best parameters, grid search (test):", clf.best_params_)
#print("Lowest RMSE, grid search (test): ", (-clf.best_score_)**(1/2.0))
#
#### Grid Search (hyperparameter tuning, train) ###
#params = { 'max_depth': [3,6,10],
#           'learning_rate': [0.01, 0.05, 0.1],
#           'n_estimators': [100, 500, 1000],
#           'colsample_bytree': [0.3, 0.7]}
#xgbr = xgb.XGBRegressor(seed = 20)
#clf = GridSearchCV(estimator=xgbr, 
#                   param_grid=params,
#                   scoring='neg_mean_squared_error', 
#                   verbose=1)
#clf.fit(X_train, y_train)
#print("Best parameters, grid search (train):", clf.best_params_)
#print("Lowest RMSE, grid search (train): ", (-clf.best_score_)**(1/2.0))

####################################################
#### Random Search (hyperparameter tuning, test) ###
#params = { 'max_depth': [3, 5, 6, 10, 15, 20],
#           'learning_rate': [0.01, 0.1, 0.2, 0.3],
#           'subsample': np.arange(0.5, 1.0, 0.1),
#           'colsample_bytree': np.arange(0.4, 1.0, 0.1),
#           'colsample_bylevel': np.arange(0.4, 1.0, 0.1),
#           'n_estimators': [100, 500, 1000]}
#xgbr = xgb.XGBRegressor(seed = 20)
#clf = RandomizedSearchCV(estimator=xgbr,
#                         param_distributions=params,
#                         scoring='neg_mean_squared_error',
#                         n_iter=25,
#                         verbose=1)
#clf.fit(X_test, y_test)
#print("Best parameters:", clf.best_params_)
#print("Lowest RMSE (Random Search, test): ", (-clf.best_score_)**(1/2.0))
#
#### Random Search (hyperparameter tuning, train) ###
#params = { 'max_depth': [3, 5, 6, 10, 15, 20],
#           'learning_rate': [0.01, 0.1, 0.2, 0.3],
#           'subsample': np.arange(0.5, 1.0, 0.1),
#           'colsample_bytree': np.arange(0.4, 1.0, 0.1),
#           'colsample_bylevel': np.arange(0.4, 1.0, 0.1),
#           'n_estimators': [100, 500, 1000]}
#xgbr = xgb.XGBRegressor(seed = 20)
#clf = RandomizedSearchCV(estimator=xgbr,
#                         param_distributions=params,
#                         scoring='neg_mean_squared_error',
#                         n_iter=25,
#                         verbose=1)
#clf.fit(X_train, y_train)
#print("Best parameters:", clf.best_params_)
#print("Lowest RMSE (Random Search, train): ", (-clf.best_score_)**(1/2.0))

#################################################
### Prediction/Inference - xgboost.Booster.predict() ###


# save model to file

print("--- dumped xgboost model ---")
print(model)

print("--- dumping 'pickle' model ---")
with open('trino.pickle.dat','wb') as f:
    pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

print("--- loading 'pickle' model ---")
with open('trino.pickle.dat','rb') as f:
    loaded_pickle_model = pickle.load(f)

print("--- loaded xgboost model ---")
print(loaded_pickle_model)

#pickle.dump(model, open("trino.pickle.dat", "wb"))
#dump(model, "trino.joblib.dat")
 
# some time later...
 
## load model from file
#loaded_model = pickle.load(open("pima.pickle.dat", "rb"))
## make predictions for test data
#y_pred = loaded_model.predict(X_test)
#predictions = [round(value) for value in y_pred]
## evaluate predictions
#accuracy = accuracy_score(y_test, predictions)
#print("Accuracy: %.2f%%" % (accuracy * 100.0))






print("=== end ===")

######################################
import sys ###########################
sys.exit() ### EXIT ##################
######################################


##############################################################
