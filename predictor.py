from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
import csv
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RepeatedKFold
import pickle

df = pd.read_csv("data.csv", encoding="ISO-8859-1", engine="python")


df = df.drop(['mths_since_last_major_derog', 'mths_since_last_delinq', 'mths_since_last_record',
              'member_id ', 'grade', 'sub_grade', 'Emp_designation', 'Experience', 'home_ownership'], axis=1)

df = df.drop(['batch_ID '], axis=1)

df1 = df[df.columns[df.isnull().any()]]
df2 = df[df.columns[~df.isnull().any()]]

df11 = df1.drop(['verification_status_joint'], axis=1)
df12 = df1[['verification_status_joint']]

# Filling missing values to numerical variables
mp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
mp_median = SimpleImputer(missing_values=np.nan, strategy='median')
df11 = mp_median.fit_transform(df11)

# filling missing values for categorical data
df12 = df12.apply(lambda x: x.fillna(x.value_counts().index[0]))

df13 = pd.DataFrame(df11)

df = pd.concat([df2, df12, df13], axis=1)

lb_make = LabelEncoder()

df['terms'] = lb_make.fit_transform(df['terms'])
df['verification_status'] = lb_make.fit_transform(df['verification_status'])
df['purpose'] = lb_make.fit_transform(df['purpose'])
df['State'] = lb_make.fit_transform(df['State'])
df['initial_list_status'] = lb_make.fit_transform(df['initial_list_status'])
df['application_type'] = lb_make.fit_transform(df['application_type'])
df['last_week_pay'] = lb_make.fit_transform(df['last_week_pay'])
df['verification_status_joint'] = lb_make.fit_transform(
    df['verification_status_joint'])

y = df[['total revol_bal']]
x = df.iloc[:, [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13,
                14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]]

scaler = StandardScaler()
x = scaler.fit_transform(x)
# a = x.columns
x = pd.DataFrame(x)
# x.columns = a


kf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=None)

for train_index, test_index in kf.split(x):
    x_train, x_test = x.iloc[train_index], x.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]


ln = LinearRegression()
lnfit = ln.fit(x_train, y_train)

pickle.dump(lnfit, open('linear.pkl', 'wb'))
