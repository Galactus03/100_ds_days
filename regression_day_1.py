""" Univariate  and multivariate regression
Day 1 of ds challange """

# Dataset used  https://www.kaggle.com/c/house-prices-advanced-regression-techniques

import numpy as np
import pandas as pd
import os
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error


import matplotlib.pyplot as plt

data_path = "./datasets/house-prices-advanced-regression-techniques"
test = pd.read_csv(os.path.join(data_path, "test.csv"))
train = pd.read_csv(os.path.join(data_path, "train.csv"))
print("shape of train data", train.shape)
print("shape of test data", test.shape)

# figuring out numeric columns and its count
numeric_columns = train._get_numeric_data()

# use this tab to plot box plot to learn about the 5 number summary and find out outliers
# plt.figure(figsize = (25,45))
# for i in enumerate(numeric_columns):
#     plt.subplot(13,3,i[0]+1)
#     sns.boxplot(train[i[1]])
#     plt.xlabel(i[1])
#     plt.show()

# we gave to remove some indexes
index = [712, 1219, 1416, 1200, 1345, 1458, 773, 1248, 1423, 628, 973, 1458, 1459]
train = train.drop(labels=index, axis=0)

# verifying the shape after
print("shape of train data", train.shape)

# figuring out the null values
Null_train = train.isnull().sum()
print(Null_train[Null_train>0])

# removing columns with too much null value
drop_columns = ["Alley","PoolQC", "Fence", "MiscFeature","Id"]
train = train.drop(drop_columns,axis=1)
test = test.drop(drop_columns,axis=1)

print("shape of train data after dropping columns with too many null values ", train.shape)
print("shape of test data after dropping columns with too many null values", test.shape)

# we will try to understand rest of the fields with null data and will replace them
Null_train_data = train[['LotFrontage', 'FireplaceQu', 'MasVnrType', 'MasVnrArea', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
      'BsmtFinType2', 'Electrical', 'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageQual', 'GarageCond']]

def analysis(data):
    return pd.DataFrame({"Data Type":data.dtypes,"Unique Count":data.apply(lambda x: x.nunique(),axis=0),
                         "Null Count":data.isnull().sum()})

# looking at what kind of data we have to figure out the mean median mode
# we replace Lotfrontage with mean and other columns with mode which is the number that occurs most times in that column
# print(Null_train_data[['LotFrontage', 'MasVnrArea', 'GarageYrBlt']].describe())

train['LotFrontage'] = train["LotFrontage"].fillna((train["LotFrontage"].mean()))
train["MasVnrArea"] = train["MasVnrArea"].fillna((train["MasVnrArea"].mode()[0]))
train["GarageYrBlt"] = train["GarageYrBlt"].fillna((train["GarageYrBlt"].mode()[0]))

# doing  the same for the test data
Null_test = test.isnull().sum()
# print(Null_test[Null_test > 0])

Null_test_data = test[['MSZoning', 'LotFrontage', 'Utilities', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'MasVnrArea',
                         'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2', 'BsmtFinSF2',
                         'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'KitchenQual', 'Functional',
                         'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageQual', 'GarageCars','GarageArea',
                         'GarageQual', 'GarageCond', 'SaleType']]
# analysing and printing the describe for test data

# print(analysis(Null_test_data))

# print(Null_test_data[['LotFrontage', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath',
#                 'BsmtHalfBath', 'GarageYrBlt', 'GarageCars', 'GarageArea']].describe())

# replacing the null valus with mean or mode as required

test['LotFrontage'] = test['LotFrontage'].fillna(test['LotFrontage'].mean())
test['MasVnrArea'] = test['MasVnrArea'].fillna(test['MasVnrArea'].mode()[0])
test['BsmtFinSF1'] = test['BsmtFinSF1'].fillna(test['BsmtFinSF1'].mode()[0])
test['BsmtFinSF2'] = test['BsmtFinSF2'].fillna(test['BsmtFinSF2'].mode()[0])
test['BsmtUnfSF'] = test['BsmtUnfSF'].fillna(test['BsmtUnfSF'].mean())
test['TotalBsmtSF'] = test['TotalBsmtSF'].fillna(test['TotalBsmtSF'].mean())
test['BsmtFullBath'] = test['BsmtFullBath'].fillna(test['BsmtFullBath'].mode()[0])
test['BsmtHalfBath'] = test['BsmtHalfBath'].fillna(test['BsmtHalfBath'].mode()[0])
test['GarageYrBlt'] = test['GarageYrBlt'].fillna(test['GarageYrBlt'].mode()[0])
test['GarageCars'] = test['GarageCars'].fillna(test['GarageCars'].mode()[0])
test['GarageArea'] = test['GarageArea'].fillna(test['GarageArea'].mean())

# figuring out correlation
def correlation(data, limit):
  col = set()
  corr_matrix = data.corr()
  for i in range(len(corr_matrix)):
    for j in range(i):
      if (corr_matrix.iloc[i, j]) > limit:
        col_name = corr_matrix.columns[i]
        col.add(col_name)
  return col

# before we start with this, let's see what a corr matrix looks like
# print(train.corr())

# now finding the correlated fields in our train data
corr_columns = correlation(train,0.7)
# print(corr_columns)

# dropping correlated fields
train = train.drop(['1stFlrSF', 'GarageArea', 'TotRmsAbvGrd'], axis = 1)
test = test.drop(['1stFlrSF', 'GarageArea', 'TotRmsAbvGrd'], axis = 1)
# print(train.head())

# time to seperate out the training value

House_Price = pd.DataFrame(train["SalePrice"])
train = train.drop(['SalePrice'], axis=1)

# let's move on the distribution side of things and plot it
sns.displot(House_Price["SalePrice"], kde=True, color='Green')
# plt.show()

# let's apply log on our target variable
house_price_log = pd.DataFrame(np.log(House_Price['SalePrice']))

# merging train and test for now to create new fields
data = pd.concat([train,test])
# print(data.shape)

# new fields created to better reprsent the data
data['YrBltRemod'] = data['YearBuilt'] + data['YearRemodAdd']
data['TotalBathrooms'] = (data['FullBath'] + (0.5 * data['HalfBath']) +
                               data['BsmtFullBath'] + (0.5 * data['BsmtHalfBath']))
data['TotalPorchSf'] = (data['OpenPorchSF'] + data['3SsnPorch'] +
                              data['EnclosedPorch'] + data['ScreenPorch'] +
                              data['WoodDeckSF'])

data["TotalOutsideSF"] = sum((data['WoodDeckSF'],data['OpenPorchSF'],data['EnclosedPorch'], data['ScreenPorch']))

data['HouseAge'] = data['YrSold'] - data['YearBuilt']

data['OverallCondQual'] = (data['OverallCond'] + data['OverallQual'])/2

# figuring out our numerical columns

data_num_cols = data._get_numeric_data().columns

# figuring out categorical values

data_cat_cols = data.columns.difference(data_num_cols)

#todo read more about _get_numeric_data being protected function of a class and what exactlt data.column.difference does

# print(data_cat_cols)

# let's seperate out both numeric and categorical data fields

data_num_data = data.loc[:, data_num_cols]
data_cat_data = data.loc[:, data_cat_cols]

print("Shape of num data:", data_num_data.shape)
print("Shape of cat data:", data_cat_data.shape)

# we ll scale the numeric data variable
#todo read more about data scaling and methods used to scale

s_scaler = StandardScaler()
data_num_data_s = s_scaler.fit_transform(data_num_data)
data_num_data_s = pd.DataFrame(data_num_data_s, columns=data_num_cols)

# converting categorical variables for training
data_cat_data = data_cat_data.fillna("NA")
label = LabelEncoder()
data_cat_data = data_cat_data.astype(str).apply(LabelEncoder().fit_transform)

# reseting index and concat for train test split
data_num_data_s.reset_index(drop=True,inplace=True)
data_cat_data.reset_index(drop=True,inplace=True)

data_new = pd.concat([data_num_data_s,data_cat_data],axis=1)

# splitting the data
#todo not sure why we are doing the steps below
train_new = data_new.loc[:1447,]
test_new = data_new.loc[1448:,]

trainx,valx,trainy,valy = train_test_split(train_new,House_Price,test_size=0.2,random_state=1234)


print(trainx.shape)
print(valx.shape)

# using linear regression
reg = LinearRegression().fit(trainx, trainy)
# print("this is linear alg score", reg.score(trainx,trainy))
# print("print coefficent and intercept",reg.coef_,reg.intercept_)


# closing notes : today most of the time was spend on prepreocessing
# we rolled with linear regresssion for now, but will spend more time tommrow on other methods and mesaure their accuracy
