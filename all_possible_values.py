import numpy as np
import pandas as pd
import csv

df = pd.read_csv("data.csv", encoding="ISO-8859-1", engine="python")

print(df['acc_now_delinq'].unique())
