import json 
import csv
import pandas as pd
import os

df = pd.read_csv('reviews.csv')
print(df.head(10))
print(df.tail(10))
