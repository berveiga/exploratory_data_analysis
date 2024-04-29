#1. Load libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import seaborn as sns
import shap
from catboost import CatBoostClassifier, Pool
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import precision_recall_curve, roc_auc_score, auc
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import PrecisionRecallDisplay, roc_curve, confusion_matrix
from utils import Params

from shapash import SmartExplainer
from lime import lime_tabular

# 2. Train random forest using the German credit dataset and randomized search

# Load parameters file
params = Params("../configuration/params_random_forest.json")

# Define parameters
cv_folds = params.cv_folds # Number of cross-validation folders

# Set seed
RF_seed = params.RF_seed
np.random.seed(params .RF_seed)

n_iter_search = params.n_iter_search # Number of iterations
all_features = True

# Extract raw credit default data from csv file
input_df_raw = pd.read_csv("../data/raw/german_processed.csv")

if all_features == False:
    features_removed = ["Gender", "PurposeOfLoan", "LoanAmount"]
# "CriticalAccountOrLoansELsewhere"
    input_df_raw = input_df_raw.drop(features_removed, axis=1)
else:
    features_removed = ["Gender", "PurposeOfLoan"]
# "CriticalAccountOrLoansELsewhere"
    input_df_raw = input_df_raw.drop(features_removed, axis=1)

input_df_raw.rename(
columns={"CriticalAccountOrLoansElsewhere": "NoHistoryOfDelayedPayments"},
inplace=True
) 

# Split into train and test data
X = input_df_raw.iloc[:, 1:]
Y = input_df_raw.iloc[:, 0]
xtrain, xtest, ytrain, ytest = train_test_split(
X, Y, test_size=0.33, random_state=RF_seed, stratify=Y
)
# Transform target values to probabilities
ytrain = ytrain.map({-1: 0, 1: 1})
ytest = ytest.map({-1: 0, 1: 1})

input_df_raw.iloc[:, 0] = input_df_raw.iloc[:, 0].map({-1: 0, 1: 1})
def nunique(x):
    #Count number of unique values
    if x is None:
        return(0)
    else:
        return(xtrain. shape[1]-(np.diff(np.sort(xtrain, axis=1), axis=1)==0).sum(1))
def get_col_indices(df, names):
    return df.columns.get_indexer(names)

nunique_xtrain = {u:xtrain[u].nunique(u) for u in xtrain.columns}
categorical_cols = [key for key, value in nunique_xtrain.items() if value <=10]
# Check how many values are assumed by each categorical feature
number_categories = {u:xtrain[u].nunique(u) for u in categorical_cols}
indices_col_features = get_col_indices(xtrain, categorical_cols)
train_pool = Pool(xtrain, ytrain, categorical_cols)
test_pool = Pool(xtest, ytest, categorical_cols)
# Instantiate model
catboost_model = CatBoostClassifier(
    custom_loss=[metrics.F1()],
    depth=2,
    random_seed=RF_seed,
    iterations=1600,
    learning_rate=1,
    verbose=True
    #Logging_Level='Silent',
)
param_grid = {
    'iterations': [100, 206, 300],
    'learning rate': [1e-3, 0.01, 0.1],
    'depth':[2, 3, 6, 8],
    '12_leaf_reg': [1, 3, 5]
}

catboost_model.grid_search(param_grid,
train_pool, plot=True)

catboost_model.fit(train_pool,
    eval_set = test_pool,
    early_stopping_rounds=5e,
    plot=True,
    silent=False)

pred_class = catboost_model.predict(Pool(xtest, ytest, categorical_cols))
probabilities catboost_train = catboost_model.predict_proba(Pool(xtrain, ytrain, categorical_cols))[:, 1]
probabilities_catboost_test = catboost_model.predict_proba(Pool(xtest, ytest, categorical_cols))[:, 1]
# Assessment of the model performance