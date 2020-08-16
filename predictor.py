from sklearn.ensemble import RandomForestRegressor
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
# df12 = df1[['verification_status_joint']]

n = df11.columns

# Filling missing values to numerical variables
mp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
mp_median = SimpleImputer(missing_values=np.nan, strategy='median')
df11 = mp_median.fit_transform(df11)

# filling missing values for categorical data
# df12 = df12.apply(lambda x: x.fillna(x.value_counts().index[0]))

df13 = pd.DataFrame(df11)

df13.columns = n

df = pd.concat([df2, df13], axis=1)

lb_make = LabelEncoder()

category_col = ['terms', 'verification_status', 'purpose', 'State', 'initial_list_status',
                'application_type']

for col in category_col:
    df[col] = lb_make.fit_transform(df[col])

y = df[['total revol_bal']]
x = df.iloc[:, [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13,
                15, 16, 17, 18, 19, 20, 21, 22, 23, 24]]

a = x.columns

scaler = StandardScaler()
x = scaler.fit_transform(x)

x = pd.DataFrame(x)
x.columns = a


# kf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=None)

# for train_index, test_index in kf.split(x):
#     x_train, x_test = x.iloc[train_index], x.iloc[test_index]
#     y_train, y_test = y.iloc[train_index], y.iloc[test_index]


ln = LinearRegression()
lnfit = ln.fit(x, y)

model = RandomForestRegressor()
model.fit(x, y)

pickle.dump(lnfit, open('linear.pkl', 'wb'))
pickle.dump(model, open('randomForest.pkl', 'wb'))
