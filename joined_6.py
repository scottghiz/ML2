#!/usr/bin/python

import boto3
import re
import s3fs
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sagemaker
import time
import datetime

### IDENTIFY SCRIPT ###
print("=============== joined_6.py ===============")

### read 'left' dataframe from csv fle ###
df0 = pd.read_csv("/home/scripts/PREDICT/TRAINING_DATA/CSV/joined_4.csv")
df1 = pd.read_csv("/home/scripts/PREDICT/TRAINING_DATA/CSV/wo_0.csv")

df2 = df0.join(df1)
del df2["wo_details_LL"]
del df2["Unnamed: 0"]
del df2["index"]
del df2["day_id_RL"]
del df2["eff_dt_RR"]

df3 = df2.drop_duplicates()
#df3 = df2

df3.columns = df3.columns.str.replace("_LL", "")
df3.columns = df3.columns.str.replace("_LR", "")
df3.columns = df3.columns.str.replace("_RL", "")
df3.columns = df3.columns.str.replace("_RR", "")

print("--- df3 ---")
print(df3)

df3.to_csv("/home/scripts/PREDICT/TRAINING_DATA/CSV/joined_6.csv")
os.system("cp /home/scripts/PREDICT/TRAINING_DATA/CSV/joined_6.csv /home/scripts/PREDICT/TRAINING_DATA/CSV/trino_ml.csv")

######################################
import sys ###########################
sys.exit() ### EXIT ##################
######################################


