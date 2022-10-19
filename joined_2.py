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
print("=============== joined_2.py ===============")

### read 'left' dataframe from csv fle ###
df0 = pd.read_csv("/home/scripts/PREDICT/TRAINING_DATA/CSV/joined_1.csv")
print("--- df0 ---")
print(df0)
df1 = df0.drop_duplicates()
drop_cols = ["Unnamed: 0","wo_ps_type_LL","JobType_LL","finalcontentcode_RL","finalintenttext_RL","fulfillmentstatus_RL","initialcontentcode_RL","initialintentcode_RL","initialintenttext_RL","Unnamed: 0_y","itg_name_RR","exit_result_RR"]

df1.drop(drop_cols, inplace=True, axis=1)
print("--- df1 ---")
print(df1)
df1.to_csv("/home/scripts/PREDICT/TRAINING_DATA/CSV/joined_2.csv")

######################################
import sys ###########################
sys.exit() ### EXIT ##################
######################################

