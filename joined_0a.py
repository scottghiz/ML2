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
print("=============== joined_0a.py ===============")

### READ IN joinfiles.txt to determine which two data tables to join ###
with open("/home/scripts/PREDICT/COMMON/joinfiles.txt") as file:
    lines = file.readlines()
    lines = [line.rstrip() for line in lines]

dict_data_table = {}
for x in range(len(lines)):
    array = lines[x].split(',')
    dict_data_table.update({array[0]: array[1]})

for k, v in dict_data_table.items():
    print(k, v)

### read 'left' dataframe from csv fle ###
dt_string = "/home/scripts/PREDICT/TRAINING_DATA/CSV/trino_out-"+dict_data_table['a0']+".csv"
print(dt_string)
df0 = pd.read_csv(dt_string)

### add '_L' to column names for clarty in resulting dataframe after merge ###
df0 = df0.add_suffix("_L")
df0['accountnumber_L'] = df0['accountnumber_L'].astype(str)
df0[df0["accountnumber_L"].str.contains("Unauthenticated")==False]

### read 'right' dataframe from csv fle ###
dt_string = "/home/scripts/PREDICT/TRAINING_DATA/CSV/trino_out-"+dict_data_table['a1']+".csv"
print(dt_string)
df1 = pd.read_csv(dt_string)

### drop columns that appear usless or problematic ###
### add '_R' to column names for clarty in resulting dataframe after merge ###
df1 = df1.add_suffix('_R')
df1['billingaccountid_R'] = df1['billingaccountid_R'].astype(str)
df1[df1["billingaccountid_R"].str.contains("Unauthenticated")==False]
print(df1)

### merge dataframes based on selected columns ###
df01 = pd.merge(df0,df1,left_on='accountnumber_L',right_on='billingaccountid_R')
print(df01)
### write merged dataframe to csv fle ###
df01.to_csv("/home/scripts/PREDICT/TRAINING_DATA/CSV/joined_0a.csv")

######################################
import sys ###########################
sys.exit() ### EXIT ##################
######################################
