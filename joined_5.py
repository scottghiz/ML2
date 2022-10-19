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
print("=============== joined_5.py ===============")

### read 'left' dataframe from csv fle ###
df0 = pd.read_csv("/home/scripts/PREDICT/TRAINING_DATA/CSV/joined_4.csv")
df1 = df0.drop_duplicates()
df2 = df1["wo_details_LL"]

print("--- df2 ---")
print(df2)

df2.to_csv("/home/scripts/PREDICT/TRAINING_DATA/CSV/joined_5.csv")

######################################
import sys ###########################
sys.exit() ### EXIT ##################
######################################

