#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from tqdm import tqdm
import pandas as pd
import numpy as np
import lightgbm as lgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb


# In[ ]:


import sys
sys.path.append('../')

from preprocess.protocol_encode import protocol2feature, load_sentence_2_vec

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[ ]:


sentence2vec = load_sentence_2_vec("../data") 


# In[ ]:


train_data = pd.read_csv(f'../data/time_prediction_train.csv', sep='\t')
test_data = pd.read_csv(f'../data/time_prediction_test.csv', sep='\t')

train_data, valid_data = train_test_split(train_data, test_size=0.2, random_state=0)
print(train_data.head())


# In[ ]:


# Missing Value Handling
train_data['criteria'].fillna('', inplace=True)
valid_data['criteria'].fillna('', inplace=True)
test_data['criteria'].fillna('', inplace=True)


# In[ ]:


# # 32 sentences length can cover 95% of the data 

# criteria_lst = train_data['criteria']

# in_criteria_lengths = []
# ex_criteria_lengths = []

# for criteria in criteria_lst:
#     in_criteria, ex_criteria = protocol2feature(criteria, sentence2vec)
#     in_criteria_lengths.append(len(in_criteria))
#     ex_criteria_lengths.append(len(ex_criteria))

# print(f"Inclusion: {pd.Series(in_criteria_lengths).describe(percentiles=[0.5, 0.75, 0.9, 0.95, 0.99, 0.999])}")
# print(f"Exclusion: {pd.Series(ex_criteria_lengths).describe(percentiles=[0.5, 0.75, 0.9, 0.95, 0.99, 0.999])}")


# In[ ]:


def criteria2embedding(criteria_lst):
    criteria_lst = [protocol2feature(criteria, sentence2vec) for criteria in criteria_lst]

    incl_criteria = []
    excl_criteria = []

    for criteria in criteria_lst:
        incl_criteria.append(torch.mean(criteria[0], dim=0))
        excl_criteria.append(torch.mean(criteria[1], dim=0))

    incl_emb = torch.stack(incl_criteria)
    excl_emb = torch.stack(excl_criteria)

    return torch.cat((incl_emb, excl_emb), dim=1)


# In[ ]:


X_train = criteria2embedding(train_data['criteria'])
X_valid = criteria2embedding(valid_data['criteria'])
X_test = criteria2embedding(test_data['criteria'])

y_train = train_data['time_day']
y_valid = valid_data['time_day']
y_test = test_data['time_day']

lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_valid, y_valid, reference=lgb_train)


# In[ ]:


# GBDT
# params = {
#     'boosting': 'gbdt',
#     'objective': 'regression',
#     'metric': 'rmse',
#     'learning_rate': 0.01,
#     'early_stopping_round': 10,
#     'verbosity': 1,
#     'max_depth': 10,
#     'num_threads': 4
# }

# RF
params = {
    'boosting': 'rf',
    'bagging_fraction': 0.8,
    'feature_fraction': 0.8,
    'objective': 'regression',
    'metric': 'rmse',
    'learning_rate': 0.1,
    'early_stopping_round': 10,
    'verbosity': 1,
    'max_depth': 10,
    'num_threads': 4
}


# In[ ]:


# Train the model
gbm = lgb.train(params, lgb_train, num_boost_round=1000, valid_sets=[lgb_eval], callbacks=[lgb.log_evaluation()])

# Predict on test data
print(gbm.best_iteration)
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
pearson_score, _ = pearsonr(y_test, y_pred)

print(f'The RMSE of prediction is: {rmse}')
print(f'The MAE of prediction is: {mae}')
print(f'The R2 of prediction is: {r2}')
print(f'The Pearson Score of prediction is: {pearson_score}')


# In[ ]:


# params = {
#     'booster': 'gbtree',
#     # 'bagging_fraction': 0.8,
#     # 'feature_fraction': 0.8,
#     'objective': 'reg:squarederror',
#     'metric': 'rmse',
#     'learning_rate': 0.1,
#     'early_stopping_round': 10,
#     'verbosity': 1,
#     # 'max_depth': 10,
#     'num_threads': 4,
#     'device':'gpu'
# }


# In[ ]:


# # Train the model
# xgb_train = xgb.DMatrix(X_train, label=y_train)
# xgb_eval = xgb.DMatrix(X_valid, label=y_valid)
# xgb_reg = xgb.train(params, xgb_train, num_boost_round=100, evals=[(xgb_eval, 'eval')], early_stopping_rounds=10)

# # Predict on test data
# print(xgb_reg.best_iteration)
# y_pred = gbm.predict(xgb.DMatrix(X_test), iteration_range=(0, gbm.best_iteration))


# # Evaluate the model
# mse = mean_squared_error(y_test, y_pred)
# rmse = np.sqrt(mse)
# mae = mean_absolute_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)
# pearson_score, _ = pearsonr(y_test, y_pred)

# print(f'The RMSE of prediction is: {rmse}')
# print(f'The MAE of prediction is: {mae}')
# print(f'The R2 of prediction is: {r2}')
# print(f'The Pearson Score of prediction is: {pearson_score}')


# In[ ]:


# # Train the model
# xgb_train = xgb.DMatrix(X_train, label=y_train)
# xgb_eval = xgb.DMatrix(X_valid, label=y_valid)
# xgb_reg = xgb.train(params, xgb_train, num_boost_round=100, evals=[(xgb_eval, 'eval')], early_stopping_rounds=10)

# # Predict on test data
# print(xgb_reg.best_iteration)
# y_pred = gbm.predict(xgb.DMatrix(X_test), iteration_range=(0, gbm.best_iteration))


# # Evaluate the model
# mse = mean_squared_error(y_test, y_pred)
# rmse = np.sqrt(mse)
# mae = mean_absolute_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)
# pearson_score, _ = pearsonr(y_test, y_pred)

# print(f'The RMSE of prediction is: {rmse}')
# print(f'The MAE of prediction is: {mae}')
# print(f'The R2 of prediction is: {r2}')
# print(f'The Pearson Score of prediction is: {pearson_score}')


# In[ ]:


# adaboost_reg = AdaBoostRegressor(n_estimators=50, learning_rate=0.5)
# adaboost_reg = adaboost_reg.fit(X_train, y_train)
# y_pred = adaboost_reg.predict(X_test)

# # Evaluate the model
# mse = mean_squared_error(y_test, y_pred)
# rmse = np.sqrt(mse)
# mae = mean_absolute_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)
# pearson_score, _ = pearsonr(y_test, y_pred)

# print(f'The RMSE of prediction is: {rmse}')
# print(f'The MAE of prediction is: {mae}')
# print(f'The R2 of prediction is: {r2}')
# print(f'The Pearson Score of prediction is: {pearson_score}')


# In[ ]:


# # Train the model
# reg = LinearRegression().fit(X_train, y_train)

# # Predict on test data
# y_pred = reg.predict(X_test)

# # Evaluate the model
# mse = mean_squared_error(y_test, y_pred)
# rmse = np.sqrt(mse)
# mae = mean_absolute_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)
# pearson_score, _ = pearsonr(y_test, y_pred)

# print(f'The RMSE of prediction is: {rmse}')
# print(f'The MAE of prediction is: {mae}')
# print(f'The R2 of prediction is: {r2}')
# print(f'The Pearson Score of prediction is: {pearson_score}')

