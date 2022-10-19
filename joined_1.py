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
print("=============== joined_1.py ===============")

### read 'left' dataframe from csv fle ###
df0 = pd.read_csv("/home/scripts/PREDICT/TRAINING_DATA/CSV/joined_0a.csv")
### add '_L' to column names for clarty in resulting dataframe after merge ###
df0 = df0.add_suffix("L")
df0['accountnumber_LL'] = df0['accountnumber_LL'].astype(str)
df0[df0["accountnumber_LL"].str.contains("Unauthenticated")==False]
df0.drop('Unnamed: 0L', inplace=True, axis=1)
print("--- df0 ---")
print(df0)
df0.to_csv("/home/scripts/PREDICT/TRAINING_DATA/CSV/tempj_0a.csv")
df00 = pd.read_csv("/home/scripts/PREDICT/TRAINING_DATA/CSV/tempj_0a.csv")

### read 'right' dataframe from csv fle ###
df1 = pd.read_csv("/home/scripts/PREDICT/TRAINING_DATA/CSV/joined_0b.csv")
### drop columns that appear usless or problematic ###
### add '_R' to column names for clarty in resulting dataframe after merge ###
df1 = df1.add_suffix('R')
df1['account_number_RR'] = df1['account_number_RR'].astype(str)
df1[df1["account_number_RR"].str.contains("Unauthenticated")==False]
df1.drop('Unnamed: 0R', inplace=True, axis=1)
print("--- df1 ---")
print(df1)
df1.to_csv("/home/scripts/PREDICT/TRAINING_DATA/CSV/tempj_0b.csv")
df11 = pd.read_csv("/home/scripts/PREDICT/TRAINING_DATA/CSV/tempj_0b.csv")

### merge dataframes based on selected columns ###
df01 = pd.merge(df00,df11,left_on='accountnumber_LL',right_on='acct_num_LR')
df01.drop('Unnamed: 0_x', inplace=True, axis=1)
print("--- df01 ---")
print(df01)
df02 = df01.drop_duplicates()
drop_cols = ["id_LL","billingaccountid_RL","customerguid_RL","querycreatetimeepoch_RL","queryid_RL","sessionid_RL","day_id_RL","acct_num_LR","testresultsid_LR","reqts_LR","day_id_LR","account_number_RR","account_context_id_RR","oracle_session_id_RR","eff_dt_RR","eff_ts_RR"]

#######################################         df02.drop(drop_cols, inplace=True, axis=1)
print("--- df02 ---")
print(df02)
### write merged dataframe to csv fle ###
df02.to_csv("/home/scripts/PREDICT/TRAINING_DATA/CSV/joined_1.csv")

######################################
import sys ###########################
sys.exit() ### EXIT ##################
######################################


