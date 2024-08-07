import pandas as pd
import seaborn as sns
import datetime as dt
import matplotlib.pyplot as plt
import lightgbm as lgb
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import pickle
import os


from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from lgbm_functions import *


### Set general parameters ----

# base directory path
filepath = "/home/unix/aargenti/"

# filepath to previous optuna params
pre_boruta_optuna_params = None

# set random seed
random_seed = 3456

# set base model parameters 
base_params = {
    "verbosity": -1,
    "objective": "regression",
    # set to none to use custom eval metric
    "metric": "None",
    # number of trees (will use less if converges)
    'n_estimators': 5000,
    # run in parallel across all available cores
    'n_jobs': -1,
    # set random state
    'random_state': random_seed
}

# number of cross-validation folds
nfolds = 5
# set number of trials for optuna
ntrials = 200
# use dart to tune
dart = True

# set date and time
now = dt.datetime.now()
now = now.strftime('%Y-%m-%d')

# analysis title
analysis = "pAge_UKB_3k_post_top_20"

### print params
print(f"File name: {analysis}", flush=True)
print(f"\nNumber of folds: {nfolds}", flush=True)
print(f"Number of Optuna trials: {ntrials}", flush=True)

### load data
with open(f'{filepath}data/UKB_data_dict_dart_2023-12-23.p', "rb") as f:
    server_data = pickle.load(f)

## load top 20 features
name = f"{filepath}results/RFE/RFECV_SHAP_top_20_features_2023-12-29.csv"
top_20_features = pd.read_csv(name, header=None)
top_20_features = list(top_20_features[0])
print(top_20_features, flush=True)


### run optuna

### Optuna hyperparameter optimization ----
engine = create_engine(f'sqlite:///{filepath}data/{analysis}.db', echo=False)
storage = f'sqlite:///{filepath}data/{analysis}.db'

# set study name
study_name = f'{analysis}_olink_params_{now}'

# create data dictionary
reduced_data = {
    'X_train': server_data['X_train'][top_20_features],
    'X_test': server_data['X_test'][top_20_features],
    'y_train': server_data['y_train'],
    'y_test': server_data['y_test']
}

# run optuna search and return the study
study_top_20 = optuna_lgbm_cv(
    base_params=base_params,
    nfolds=nfolds,
    ntrials=ntrials,
    random_seed=random_seed,
    data_dict=reduced_data,
    multi_opt=False,
    scoring_metric='r2_score',
    study_name=study_name,
    storage=storage,
    ensemble=False,
    prune=False,
    dart = True
)

# get best parameters
best_params_top_20 = study_top_20.best_params

# save final params
filename = f'{filepath}output/optuna/{analysis}_optuna_best_params_post_top_20_{now}.p'
with open(filename, "wb") as f:
    pickle.dump(best_params_top_20, f, protocol=4)

# save study
filename = f'{filepath}output/optuna/{analysis}_optuna_study_post_top_20_{now}.p'
with open(filename, "wb") as f:
    pickle.dump(study_top_20, f, protocol=4)
