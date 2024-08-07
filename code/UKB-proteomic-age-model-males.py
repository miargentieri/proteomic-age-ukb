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
    "boosting_type": "gbdt",
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
# set number of trials for boruta
boruta_trials = 200
# set boruta threshold used to compare shadow and real features 
perc = 100
# do I run boruta?
boruta = True
# add cohort weights
weights_eval = False
# minMax scal data
scale = True
# tune dart?
dart = True

# set date and time
now = dt.datetime.now()
now = now.strftime('%Y-%m-%d')

# analysis title
analysis = "pAge_UKB_3k_men_dart"
if weights_eval:
    analysis = f"{analysis}_weights"
if scale:
    analysis = f"{analysis}_minMax"


### print params
print(f"File name: {analysis}", flush=True)
print(f"\nNumber of folds: {nfolds}", flush=True)
print(f"Number of Optuna trials: {ntrials}", flush=True)
print(f"Boruta threshold: {perc}", flush=True)
print(f"Boruta will be run: {boruta}", flush=True)
print(f"Cohort weights added: {weights_eval}", flush=True)
print(f"Proteins will be scaled: {scale}", flush=True)

# make sub-directories

PATH = f'{filepath}output'
if not os.path.exists(PATH):
    os.makedirs(PATH)
PATH = f'{filepath}output/final_model'
if not os.path.exists(PATH):
    os.makedirs(PATH)
PATH = f'{filepath}output/final_score'
if not os.path.exists(PATH):
    os.makedirs(PATH)
PATH = f'{filepath}output/model_metrics/'
if not os.path.exists(PATH):
    os.makedirs(PATH)
PATH = f'{filepath}output/optuna'
if not os.path.exists(PATH):
    os.makedirs(PATH)  
PATH = f'{filepath}output/shap'
if not os.path.exists(PATH):
    os.makedirs(PATH)  
   
PATH = f'{filepath}results'
if not os.path.exists(PATH):
    os.makedirs(PATH)
PATH = f'{filepath}results/boruta'
if not os.path.exists(PATH):
    os.makedirs(PATH)
PATH = f'{filepath}results/single_model'
if not os.path.exists(PATH):
    os.makedirs(PATH)
PATH = f'{filepath}results/final_model'
if not os.path.exists(PATH):
    os.makedirs(PATH)

# names of olink proteins in data
list_path = f'{filepath}data/olink_names_oct_30_2023.csv'
olink_names = pd.read_csv(list_path, header=None)
olink_names = list(olink_names[0])

remove_prots = [
    # proteins in UKB but not CKB
    'HLA_A',
    'ERVV_1',
    
    # proteins in CKB but not UKB
    'CD97',
    'FGFR1OP',
    'LRMP',
    'CASC4',
    'DARS',
    'HARS',
    'WISP2',
    'FOPNL',
    'WISP1',
    
    # proteins not in FinnGen
    'EDEM2',
    'EP300',
    'CGA',
    'CDHR1',
    'CPLX2',
    'CLSTN1',
    'IFIT1',
    'FGF3',
    'TAGLN3',
    'YAP1',
    'ADIPOQ',
    'BCL2L11',
    'BMP6',
    'BID',
    'SH3GL3',
    'ARL13B',
    'ANGPTL7',
    'MGLL',
    'MPI',
    'MAGEA3',
    'KCNH2'
]

extra = [
    # proteins missing > 20%
    'GLIPR1', 
    'NPM1', 
    'PCOLCE' 
]

combined = remove_prots + extra

# remove proteins not common to both cohorts
olink_names = [prot for prot in olink_names if prot not in combined]

# remove proteins not common to both cohorts
olink_names = [prot for prot in olink_names if prot not in remove_prots]

### UKB Data prep ----

# path to data
data_path = f'{filepath}data/olink_data_imputed_nov_18_2023.csv'

# load data
data = pd.read_csv(data_path)

# load age data
age_path = f'{filepath}data/granular_age_june_22_2023.csv'
age_data = pd.read_csv(age_path)

# merge
data = pd.merge(data, age_data, on='eid', how='inner')

# set index to be eid
data.set_index('eid', inplace=True)

# keep only necessary columns
keep_cols = ['age_granular', 'olink_batch'] + olink_names
data = data[keep_cols]

# load eids to be removed
eid_path = f'{filepath}data/remove_eids_61054_2023-04-25.csv'
eids = pd.read_csv(eid_path, header=None)
eids = eids.iloc[:, 0].tolist()

# remove
data = data[~data.index.isin(eids)]

#### minMax scaling ----
if scale:
    # Initialize scaling function
    scaler = MinMaxScaler()

    for protein in olink_names:
        # Extract the values for the current protein
        protein_values = data[protein].values.reshape(-1, 1).copy()
        # Fit and transform the data for the current protein
        scaled_values = scaler.fit_transform(protein_values)
        # Calculate the median for the current protein
        median = np.nanmedian(scaled_values)
        # Median center the current protein
        centered_values = scaled_values - median
        # Add the scaled and centered data to the new DataFrame
        data[protein] = centered_values

# check if scale was done:
print(f'\nAPOE distribution in UKB:\n{data["APOE"].describe()}', flush=True)


# remove those with NA for batch
data = data[data['olink_batch'].notna()]

# remove those not in random subsample
data = data[~data['olink_batch'].isin((0, 7))]

# import and merge sex data
exposure_path = f'{filepath}data/ukb_imputation1_jul_25_2023.feather'
exposure_data = pd.read_feather(exposure_path)
exposure_data = exposure_data[['eid','sex']]
exposure_data.set_index('eid', inplace=True)
data = pd.merge(data, exposure_data, left_index=True, right_index=True, how='inner')

# subset to just men
data = data[data['sex'] == 'Male']

# remove rows (participants) missing values for >20% of proteins
data = data[data.isna().sum(axis=1) <= 0.20 * data[olink_names].shape[1]]

# get names of columns missing >20%
prot_miss = data.columns[data.isna().sum(axis=0) > 0.20 * data[olink_names].shape[0]]
prot_miss = list(prot_miss)
# map to list of protein columns
prot_miss = [prot for prot in prot_miss if prot in olink_names]
print(f'\nProteins with >20% missing in UKB: {prot_miss}', flush=True)

# remove proteins with more than 20% missing
olink_names = [prot for prot in olink_names if prot not in prot_miss]

print("\nNumber of proteins tested:", len(olink_names), flush=True)
print('APOE is in proteins queried:', 'APOE' in olink_names, flush=True)
print('FOXO3 is in proteins queried:', 'FOXO3' in olink_names, flush=True)


# round age to be same number of decimals as CKB
data['age_granular'] = data['age_granular'].round(2)


# split X and y
X_ukb = data[olink_names]
y_ukb = data['age_granular']

# split UKB data for training/testing 
X_train_ukb, X_test_ukb, y_train_ukb, y_test_ukb = train_test_split(
    X_ukb, 
    y_ukb,
    train_size=0.7,
    random_state=random_seed,
    shuffle=True
) 

# create data dictionary
data_dict = {
    'X_train': X_train_ukb,
    'X_test': X_test_ukb,
    'y_train': y_train_ukb,
    'y_test': y_test_ukb
}

# save to use exact data later
filename = f'{filepath}results/UKB_data_dict_men_{now}.p'
with open(filename, "wb") as f:
    pickle.dump(data_dict, f, protocol=4)



### CKB data prep ----

# path to data
data_path = f'{filepath}data/ckb_coded_olink_oct_17_2023.csv'

# load data
ckb_data = pd.read_csv(data_path)

# set index to be csid
ckb_data.set_index('csid', inplace=True)

# rename column to match UKB
ckb_data['age_granular'] = ckb_data['recruitment_age'].copy()

# print summary of age in CKB subset
print(f'\nAge distribution in CKB:\n{ckb_data["age_granular"].describe()}')

#### minMax scaling ----
if scale:
    # Initialize scaling function
    scaler = MinMaxScaler()

    for protein in olink_names:
        # Extract the values for the current protein
        protein_values = ckb_data[protein].values.reshape(-1, 1).copy()
        # Fit and transform the data for the current protein
        scaled_values = scaler.fit_transform(protein_values)
        # Calculate the median for the current protein
        median = np.nanmedian(scaled_values)
        # Median center the current protein
        centered_values = scaled_values - median
        # Add the scaled and centered data to the new DataFrame
        ckb_data[protein] = centered_values
    
# check if scale was done:
print(f'\nAPOE distribution in CKB:\n{ckb_data["APOE"].describe()}', flush=True)

# subet to just men
ckb_data = ckb_data[ckb_data['is_female'] == 0]

# duplicate
ckb_data_all = ckb_data.copy()

# subset to those in random subset
ckb_data_random = ckb_data[ckb_data['olinkexp1536_chd_b1_subcohort'] == 1].copy()

# set X, y
X_ckb = ckb_data_random[olink_names]
y_ckb = ckb_data_random['age_granular']

X_ckb_all = ckb_data_all[olink_names]
y_ckb_all = ckb_data_all['age_granular']


# check sample size
print("\nUKB sample size:", len(data.index), flush=True)
print("CKB sample size (random):", len(ckb_data_random.index), flush=True)
print("CKB sample size (all):", len(ckb_data_all.index), flush=True)

print("UKB train size:", len(X_train_ukb.index), flush=True)
print("UKB test size:", len(X_test_ukb.index), flush=True)


#### Run Boruta pipeline ----

# check if sql database exists from previous analysis. if so, delete
PATH = f'{filepath}data/{analysis}.db'
PATH2 = f'{filepath}data/{analysis}_post_boruta.db'

if os.path.exists(PATH):
    os.remove(PATH)
if os.path.exists(PATH2):
    os.remove(PATH2)

results = lgbm_pipeline_regression(
    base_params=base_params,
    ntrials=ntrials, 
    boruta_trials=boruta_trials,
    nfolds=nfolds, 
    random_seed=random_seed, 
    data_dict=data_dict, 
    analysis=analysis,
    filepath=filepath, 
    perc=perc,
    scoring_metric='r2_score',
    boruta=boruta,
    sample_weight=None,
    eval_sample_weight=None,
    pre_boruta_optuna_params = pre_boruta_optuna_params,
    dart = dart
)

# get best parameters from optuna
best_params = results.best_params

# subset datasets to selected features
if boruta:
    selected_features = results.selected_features
    X_ukb = X_ukb[selected_features]
    X_ckb = X_ckb[selected_features]
    X_ckb_all = X_ckb_all[selected_features]


### Evaluation against CKB independent data (random samples) ----

# update tuned parameters with base parameters
best_params.update(base_params)

# Intialize model
model = lgb.LGBMRegressor(**best_params)

# set callbacks
callbacks = [lgb.log_evaluation(period=1), lgb.early_stopping(20, verbose=False)]

# fit model
model.fit(
    X=data_dict['X_train'][selected_features], 
    y=data_dict['y_train'],  
    eval_set=[(X_ckb, y_ckb)],
    eval_metric=r2_score_lgbm,
    callbacks=callbacks,
    sample_weight=None
)

# evaluation metics
predictions = model.predict(X_ckb)
r2_pre = r2_score(y_ckb, predictions)
r2_ci_pre = list(rsquareCI(y_ckb, predictions, X_ckb, 0.95))

# eval metrics
print(f"R squared in CKB test data (random sample): {r2_pre}, 95% CI: {r2_ci_pre[0]:.4f}-{r2_ci_pre[1]:.4f}", flush=True)

# make df of predicted age
pred_df_ckb = pd.DataFrame(index=X_ckb.index)
pred_df_ckb['predicted_values'] = predictions

# merge with y
pred_df_ckb = pd.merge(pred_df_ckb, y_ckb, left_index=True, right_index=True)

# make column for csid
pred_df_ckb['csid'] = pred_df_ckb.index



### Evaluation against CKB independent data (all samples) ----

# Intialize model
model = lgb.LGBMRegressor(**best_params)

# set callbacks
callbacks = [lgb.log_evaluation(period=1), lgb.early_stopping(20, verbose=False)]

# fit model
model.fit(
    X=data_dict['X_train'][selected_features], 
    y=data_dict['y_train'],  
    eval_set=[(X_ckb_all, y_ckb_all)],
    eval_metric=r2_score_lgbm,
    callbacks=callbacks,
    sample_weight=None
)

# evaluation metics
predictions = model.predict(X_ckb_all)
r2_pre = r2_score(y_ckb_all, predictions)
r2_ci_pre = list(rsquareCI(y_ckb_all, predictions, X_ckb_all, 0.95))

# eval metrics
print(f"R squared in CKB test data (all participants): {r2_pre}, 95% CI: {r2_ci_pre[0]:.4f}-{r2_ci_pre[1]:.4f}", flush=True)

# make df of predicted age
pred_df_ckb_all = pd.DataFrame(index=X_ckb_all.index)
pred_df_ckb_all['predicted_values'] = predictions

# merge with y
pred_df_ckb_all = pd.merge(pred_df_ckb_all, y_ckb_all, left_index=True, right_index=True)

# make column for csid
pred_df_ckb_all['csid'] = pred_df_ckb_all.index


#### Predictions on all the UKB data ----

# update parameters
best_params.update(base_params)

# new data dictionary with unsplit data
full_data = {}
full_data = {
    'X_train': X_ukb,
    'y_train': y_ukb
}

# get predictions on all data using cross-validation
r2, pred_df_ukb = lgbm_cv_regression(
    nfolds=nfolds, 
    random_seed=random_seed, 
    data_dict=full_data, 
    params=best_params,
    weights=None
)

# R2 across folds
print(f'R squared across folds from all UKB data: {r2}')

# merge with y
pred_df_ukb = pd.merge(pred_df_ukb, y_ukb, left_index=True, right_index=True)

# make columns for eids
pred_df_ukb['eid'] = pred_df_ukb.index




#### UKB test set plot --------

# Intialize model
model = lgb.LGBMRegressor(**best_params)

# set callbacks
callbacks = [lgb.log_evaluation(period=1), lgb.early_stopping(20, verbose=False)]

# fit model
model.fit(
    X=data_dict['X_train'][selected_features], 
    y=data_dict['y_train'],  
    eval_set=[(data_dict['X_test'][selected_features], data_dict['y_test'])],
    eval_metric=r2_score_lgbm,
    callbacks=callbacks,
    sample_weight=None
)

# evaluation metics
predictions = model.predict(data_dict['X_test'][selected_features])
r, pvalue = pearsonr(data_dict['y_test'], predictions)
r2 = r2_score(data_dict['y_test'], predictions)
rmse = mean_squared_error(data_dict['y_test'], predictions, squared=False)
mae = mean_absolute_error(data_dict['y_test'], predictions)

# plot predicted age against age
regplot = sns.regplot(
    x=predictions, 
    y=data_dict['y_test'],
    scatter_kws=dict(color='midnightblue', s=10, alpha=0.8),
    line_kws=dict(color='red')
)

# add annotation
annotation_text = f'r = {r:.2f}\nR² = {r2:.2f}\nRMSE = {rmse:.2f}\nMAE = {mae:.2f}'
plt.text(.01, .99, annotation_text, ha='left', va='top', transform=regplot.transAxes)
plt.text(.95, .1, 'UKB test set', ha='right', va='top', transform=regplot.transAxes)


# p-value = {pvalue:.2e} 
regplot.set(xlabel='Age', ylabel='ProtAge')
fig = regplot.get_figure()

# save
name = f'{filepath}output/final_model/{analysis}_plot_UKB_{now}.png'
fig.savefig(
    name,
    dpi=600,
    transparent=False,
    bbox_inches="tight"
) 
plt.close()

## CKB plot (random subset)
r, pvalue = pearsonr(pred_df_ckb['age_granular'], pred_df_ckb['predicted_values'])
r2 = r2_score(pred_df_ckb['age_granular'], pred_df_ckb['predicted_values'])
rmse = mean_squared_error(pred_df_ckb['age_granular'], pred_df_ckb['predicted_values'], squared=False)
mae = mean_absolute_error(pred_df_ckb['age_granular'], pred_df_ckb['predicted_values'])

# plot predicted age against age
regplot = sns.regplot(
    data=pred_df_ckb,
    x='age_granular', 
    y='predicted_values',
    scatter_kws=dict(color='midnightblue', s=10, alpha=0.8),
    line_kws=dict(color='red')
)

# add annotation
annotation_text = f'r = {r:.2f}\nR² = {r2:.2f}\nRMSE = {rmse:.2f}\nMAE = {mae:.2f}'
plt.text(.01, .99, annotation_text, ha='left', va='top', transform=regplot.transAxes)
plt.text(.95, .1, 'CKB', ha='right', va='top', transform=regplot.transAxes)

# p-value = {pvalue:.2e} 
regplot.set(xlabel='Age', ylabel='ProtAge')
fig = regplot.get_figure()

# save
name = f'{filepath}output/final_model/{analysis}_plot_CKB_random_{now}.png'
fig.savefig(
    name,
    dpi=600,
    transparent=False,
    bbox_inches="tight"
) 
plt.close()

## CKB plot (all participants)
r, pvalue = pearsonr(pred_df_ckb_all['age_granular'], pred_df_ckb_all['predicted_values'])
r2 = r2_score(pred_df_ckb_all['age_granular'], pred_df_ckb_all['predicted_values'])
rmse = mean_squared_error(pred_df_ckb_all['age_granular'], pred_df_ckb_all['predicted_values'], squared=False)
mae = mean_absolute_error(pred_df_ckb_all['age_granular'], pred_df_ckb_all['predicted_values'])

# plot predicted age against age
regplot = sns.regplot(
    data=pred_df_ckb_all,
    x='age_granular', 
    y='predicted_values',
    scatter_kws=dict(color='midnightblue', s=10, alpha=0.8),
    line_kws=dict(color='red')
)

# add annotation
annotation_text = f'r = {r:.2f}\nR² = {r2:.2f}\nRMSE = {rmse:.2f}\nMAE = {mae:.2f}'
plt.text(.01, .99, annotation_text, ha='left', va='top', transform=regplot.transAxes)
plt.text(.95, .1, 'CKB', ha='right', va='top', transform=regplot.transAxes)

# p-value = {pvalue:.2e} 
regplot.set(xlabel='Age', ylabel='ProtAge')
fig = regplot.get_figure()

# save
name = f'{filepath}output/final_model/{analysis}_plot_CKB_all_{now}.png'
fig.savefig(
    name,
    dpi=600,
    transparent=False,
    bbox_inches="tight"
) 
plt.close()


# save to file
name = f'{filepath}output/final_score/{analysis}_scores_UKB_{now}.csv'
pred_df_ukb.to_csv(name, index=False)

name = f'{filepath}output/final_score/{analysis}_scores_CKB_random_{now}.csv'
pred_df_ckb.to_csv(name, index=False)

name = f'{filepath}output/final_score/{analysis}_scores_CKB_all_{now}.csv'
pred_df_ckb_all.to_csv(name, index=False)
