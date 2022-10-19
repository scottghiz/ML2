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
print("=============== joined_3.py ===============")

### read 'left' dataframe from csv fle ###
df0 = pd.read_csv("/home/scripts/PREDICT/TRAINING_DATA/CSV/joined_2.csv")
print("--- df0 ---")
print(df0)
# array([ True, False, False, False])
df1 = df0.loc[:,~df0.columns.str.startswith('Unnamed')]
df2 = df1.drop_duplicates()
print("--- df2 ---")
print(df2)
df2.to_csv("/home/scripts/PREDICT/TRAINING_DATA/CSV/joined_3.csv")

######################################
import sys ###########################
sys.exit() ### EXIT ##################
######################################

