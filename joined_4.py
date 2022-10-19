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
print("=============== joined_4.py ===============")

### read 'left' dataframe from csv fle ###
df0 = pd.read_csv("/home/scripts/PREDICT/TRAINING_DATA/CSV/joined_3.csv")
del df0["id_LL"]
del df0["billingaccountid_RL"]
del df0["acct_num_LR"]
del df0["account_number_RR"]
df1 = df0.loc[:,~df0.columns.str.startswith('Unnamed')]
print("--- df1 ---")
print(df1)
df1.to_csv("/home/scripts/PREDICT/TRAINING_DATA/CSV/joined_4.csv")

######################################
import sys ###########################
sys.exit() ### EXIT ##################
######################################

