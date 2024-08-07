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

# set random seed
random_seed = 3456
# set global random seed
random.seed(random_seed)
np.random.seed(random_seed)

# base directory path
filepath = "/home/unix/aargenti/"

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

# set date and time
now = dt.datetime.now()
now = now.strftime('%Y-%m-%d')

### load data
with open(f'{filepath}data/UKB_data_dict_dart_2023-12-23.p', "rb") as f:
    server_data = pickle.load(f)

## load Boruta selected features
name = f"{filepath}results/boruta/pAge_UKB_3k_dart_minMax_Boruta_vars_to_keep_2023-12-23.csv"
selected_features = pd.read_csv(name, header=None)
selected_features = list(selected_features[0])
print(selected_features, flush=True)


### RFECV with SHAP

# load best params
with open(f'{filepath}output/optuna/pAge_UKB_3k_dart_minMax_optuna_best_params_post_boruta_2023-12-23.p', "rb") as f:
    tuned_params = pickle.load(f)

### Set general parameters ----

# create data dictionary
post_boruta_dict = {
    'X_train': server_data['X_train'][selected_features],
    'X_test': server_data['X_test'][selected_features],
    'y_train': server_data['y_train'],
    'y_test': server_data['y_test']
}

# set random seed
random_seed = 3456

# set base model parameters 
base_params = {
    "verbosity": -1,
    "objective": "regression",
    # 'max_depth': 1,
    # set to none to use custom eval metric
    # "metric": "rmse",
    "metric": "None",
    # "boosting_type": "gbdt",
    # number of trees (will use less if converges)
    'n_estimators': 5000,
    # run in parallel across all available cores
    'n_jobs': -1,
    # set random state
    'random_state': random_seed
}
# update tuned params
tuned_params.update(base_params)

# Minimum number of features to consider
min_features = 5

rfecv_shap = RFECV_shap(
    nfolds=nfolds, 
    random_seed=random_seed, 
    data_dict=post_boruta_dict, 
    params=tuned_params, 
    min_features=min_features
)

# save study
filename = f'{filepath}results/RFE/RFECV_SHAP_model_post_boruta_{now}.p'
with open(filename, "wb") as f:
    pickle.dump(rfecv_shap, f, protocol=4)

# get proteins from 20 protein model
top_20_features = rfecv_shap.features['20']

# save selected features 
filename = f'{filepath}results/RFE/RFECV_SHAP_top_20_features_{now}.csv'
pd.Series(top_20_features).to_csv(
    filename,
    header=False,
    index=False
)

# Create a DataFrame from the dictionary
df = pd.DataFrame(list(rfecv_shap.scores.items()), columns=['Number_proteins', 'R2'])

# sort df
df['Number_proteins'] = df['Number_proteins'].astype(int)
df = df.sort_values(by='Number_proteins', ascending=True)

# plot
# plt.figure()
plt.figure(figsize=(10, 5))

plt.xlabel("Number of proteins")
plt.ylabel("Mean R² across folds")
plt.plot(
    df['Number_proteins'], 
    df['R2'], 
    marker='o', 
    linestyle='-', 
    color='b', 
    markersize=2
)

# add line for 20 proteins
plt.vlines(20, ymin=(min(df['R2'])), ymax=(max(df['R2'])), linestyles='dashed', colors='gray')

# set axis ticks
plt.xticks(np.arange(0, 210, 10)) 

# title
plt.title("Recursive Feature Elimination Using SHAP Values")

name = f'{filepath}results/RFE/RFE_with_SHAP.png'
plt.savefig(
    name,
    dpi=600,
    facecolor='white',
    transparent=False,
    bbox_inches="tight"
)
plt.close()