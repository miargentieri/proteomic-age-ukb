import lightgbm as lgb
import xgboost as xgb
import optuna
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import plotly.io as pio
import shap
import pickle
import random
import os
import multiprocessing

from collections import Counter
from sqlalchemy import create_engine
from shaphypetune import BoostBoruta
from scipy import special, optimize
from numpy import interp
from optuna.samplers import TPESampler, NSGAIISampler
from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState
from sklearn.model_selection import train_test_split
from sklearn.model_selection import (KFold, StratifiedKFold)
from sklearn.metrics import roc_auc_score, roc_curve, auc, mean_squared_error, f1_score, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, precision_recall_curve
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.base import is_classifier
from sklearn import metrics
from sklearn import model_selection
from math import sqrt
from joblib import Parallel, delayed, parallel_backend

class BorutaClassification:
    '''
    Boruta using a classification model
    '''
    def __init__(self, best_params_post, selected_features, df, eval_df, eval_df_post, pred_df_post, pred_df_test):
        self.best_params = best_params_post
        self.selected_features = selected_features
        self.summary = df
        self.test_eval_pre = eval_df
        self.test_eval_post = eval_df_post
        self.train_preds = pred_df_post
        self.test_preds = pred_df_test

class LGBMClassification:
    '''
    LightGBM using a classification model
    '''
    def __init__(self, best_params, df):
        self.best_params = best_params
        self.summary = df

class BorutaRegression:
    '''
    Boruta using a regression model
    '''
    def __init__(self, best_params_post, selected_features, df):
        self.best_params = best_params_post
        self.selected_features = selected_features
        self.summary = df

class LGBMRegression:
    '''
    Boruta using a regression model
    '''
    def __init__(self, best_params, df):
        self.best_params = best_params
        self.summary = df


class RFEshap_test:
    '''
    RFE using CV and SHAPS
    '''
    def __init__(self, features, scores):
        self.features = features
        self.scores = scores

def RFECV_shap(
    nfolds:int, 
    random_seed:int, 
    data_dict:dict, 
    params:dict, 
    categorical_features:list = None,
    min_features:int = None
    ):
    
    # initialize stratified fold split
    folds = KFold(
        n_splits=nfolds, 
        shuffle=True,
        random_state=random_seed
    )
    
    # initialize list to document metrics across numbers of features
    metric_dict = {}
    features_dict = {}
    
    features = list(data_dict['X_train'].columns)
    
    for i, vars in enumerate(features):
        
        # set features list
        if i == 0:
            features = features
            features_dict.update({f'i': features})
        else:
            features = [x for x in features if x != drop_feature]
            features_dict.update({f'{len(features)}': features})
        
        if len(features) < min_features:
            break
        
        # update data dict with features for this round
        data_dict.update({'X_train': data_dict['X_train'][features], 'X_test': data_dict['X_test'][features]})
        
        # print progress
        print(f'\rNumber of features in current model: {len(features)}', flush=True, end='\r')

        # initialize list to document metrics across folds
        scores = []
        drop_features = []
        
        # split train data into folds, fit model within each fold
        for i, (train_idxs, val_idxs) in enumerate(folds.split(data_dict['X_train'], data_dict['y_train'])):
            eval_data = {
                'X_val':data_dict['X_train'].iloc[val_idxs], 
                'y_val':data_dict['y_train'].iloc[val_idxs],
                'X_train':data_dict['X_train'].iloc[train_idxs], 
                'y_train':data_dict['y_train'].iloc[train_idxs]
            }
            
            # initialize model
            model = lgb.LGBMRegressor(**params)

            # set callback
            if params['boosting_type'] == 'gbdt':
                callbacks = [lgb.log_evaluation(period=100), lgb.early_stopping(20, verbose=False)]
            elif params['boosting_type'] == 'dart':
                callbacks = None

            # set categorical features, if any
            if categorical_features is not None:
                # fit model in train data
                model.fit(
                    X=eval_data['X_train'],
                    y=eval_data['y_train'],
                    eval_set=(eval_data['X_val'], eval_data['y_val']),
                    eval_metric=r2_score_lgbm,
                    categorical_feature=categorical_features,
                    callbacks=callbacks
                )
            else:
                # fit model in train data
                model.fit(
                    X=eval_data['X_train'],
                    y=eval_data['y_train'],
                    eval_set=(eval_data['X_val'], eval_data['y_val']),
                    eval_metric=r2_score_lgbm,
                    callbacks=callbacks
                )

            # get predicted values
            preds = model.predict(eval_data['X_val'])

            # get R squared in the fold
            r2 = r2_score(eval_data['y_val'], preds)

            # append to list of R squared across folds
            scores.append(r2)
            
            # get shap values
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(eval_data['X_train'])
            
            # aggregate SHAP values for each feature
            feature_importance = np.abs(shap_values).mean(axis=0)

            # sort features based on importance scores
            sorted_features = pd.Series(
                feature_importance, 
                index=eval_data['X_train'].columns
            ).sort_values(ascending=False)

            # select the worst feature
            drop_features.append(sorted_features.index[-1])
            
        # get average R squared across folds
        avg_score = np.mean(scores)
        # update dict of metrics
        metric_dict.update({f'{len(features)}': avg_score})
        # get most common feature ranked last
        element_counts = Counter(drop_features)
        drop_feature = max(element_counts, key=element_counts.get)
        

    # make class object
    results = RFEshap_test(
        features_dict, 
        metric_dict
    )
    
    return results



def ensemble_learn(
    data_dict:dict,
    params:dict,
    ratio:float,
    scoring_metric:str,
    score1:str = None,
    score2:str = None,
    unbalanced:bool = False,
    categorical_features:list = None,
    max_fpr:float = 0.05,
    threshold:float = 0.50,
    multi_opt:bool = False,
    sample_weight:str = None,
    eval_sample_weight:str = None,
    ):
    
    # dictionary of scoring metric functions
    evals = {
        'f1_score': f1_score_lgbm,
        'pr_auc': pr_auc_lgbm,
        'precision': precision_lgbm,
        'partial_auc': partial_auc_lgbm,
        'r2_score': r2_score_lgbm,
        'auc': 'auc'
    }
    
    # make list of metrics if multiple optimzation metrics
    if multi_opt:
        metric = [evals[score1], evals[score2]]
    else:
        metric = evals[scoring_metric]
    
    # Identify minority and majority class samples
    minority_samples = data_dict['X_train'][data_dict['y_train'] == 1]
    majority_samples = data_dict['X_train'][data_dict['y_train'] == 0]

    # Full index list of all majority samples
    majority_indices = majority_samples.index.tolist()
    # Randomly shuffle index list
    np.random.shuffle(majority_indices)

    # Count length of minority and majority samples
    num_minority_samples = len(minority_samples)
    num_majority_samples = len(majority_samples)
    
    # Set number of ensemble models
    ensemble_size = num_majority_samples // int(ratio * num_minority_samples)
    
    # Initialize index counter
    majority_samples_used = 0
    
    # List to store index for sampling in each ensemble member
    indices_list = []
    # List to store predicted probabilities from each ensemble member
    scores = []
    
    # Class weights
    if unbalanced:
        params.update({'is_unbalance': 'True'})
    
    # Make list of index values to subset majority class in each ensemble member
    for member in range(ensemble_size):
        # Determine the subset of majority class samples to use
        subset_indices = majority_indices[majority_samples_used: majority_samples_used + int(ratio * num_minority_samples)]
        # Reset start of index for next iteration
        majority_samples_used += int(ratio * num_minority_samples)

        selected_majority_samples = majority_samples.loc[subset_indices]

        # Combine minority and selected majority samples
        X_train_balanced = pd.concat([minority_samples, selected_majority_samples], axis=0)
        y_train_balanced = pd.Series([1] * num_minority_samples + [0] * int(ratio * num_minority_samples))
        
        # Initialize model
        model = lgb.LGBMClassifier(**params)

        # callbacks (early stopping)
        if params['boosting_type'] == 'gbdt':
            callbacks = [
                # lgb.log_evaluation(period=100), 
                lgb.early_stopping(20, verbose=False)
            ]
        elif params['boosting_type'] == 'dart':
            callbacks = None

        # Fit model
        if categorical_features is not None:
            model.fit(
                X=X_train_balanced,
                y=y_train_balanced,
                eval_set=(data_dict['X_test'], data_dict['y_test']),
                eval_metric=scoring_metric,
                categorical_feature=categorical_features,
                callbacks=callbacks,
                sample_weight=sample_weight,
                eval_sample_weight=eval_sample_weight
            )
        else:
            model.fit(
                X=X_train_balanced,
                y=y_train_balanced,
                eval_set=(data_dict['X_test'], data_dict['y_test']),
                eval_metric=scoring_metric,
                callbacks=callbacks,
                sample_weight=sample_weight,
                eval_sample_weight=eval_sample_weight
            )

        # get predicted probabilities from ensemble model
        preds = model.predict_proba(data_dict['X_test'])[:, 1]
        
        # update list of predicted probabilities across ensemble members
        scores.append(preds)
        
        print(f'completed ensemble {member+1}/{ensemble_size}', flush=True, end='\r')
    
    # Average the predicted probabilities from all ensemble models
    avg_preds = np.mean(scores, axis=0)
    
    # Convert averaged predicted probabilities into binary class labels
    predictions = np.where(avg_preds >= threshold, 1, 0)
    
    # Apply the inverse of the sigmoid function (logit transformation)
    raw_scores = np.log(avg_preds / (1 - avg_preds))
            
    # Scoring metrics
    f1 = f1_score(data_dict['y_test'], predictions)
    precision, recall, thresholds = precision_recall_curve(data_dict['y_test'], avg_preds)
    pr_auc = auc(recall, precision)
    partial_auc = roc_auc_score(data_dict['y_test'], avg_preds, max_fpr=max_fpr)
    roc_auc = roc_auc_score(data_dict['y_test'], avg_preds)
    p_score = precision_score(data_dict['y_test'], predictions)
    r_score = recall_score(data_dict['y_test'], predictions)

    metrics = {
        'f1_score': f1,
        'pr_auc': pr_auc,
        'partial_auc': partial_auc,
        'roc_auc': roc_auc,
        'precision': p_score,
        'recall': r_score,
        'predicted_probabilities': avg_preds,
        'predictions': predictions,
        'raw_scores': raw_scores
    }

    return metrics
     
# calculate 95% CI of AUC
def roc_auc_ci(y_true, y_score, positive=1):
    AUC = roc_auc_score(y_true, y_score)
    N1 = sum(y_true == positive)
    N2 = sum(y_true != positive)
    Q1 = AUC / (2 - AUC)
    Q2 = 2*AUC**2 / (1 + AUC)
    SE_AUC = sqrt((AUC*(1 - AUC) + (N1 - 1)*(Q1 - AUC**2) + (N2 - 1)*(Q2 - AUC**2)) / (N1*N2))
    lower = AUC - 1.96*SE_AUC
    upper = AUC + 1.96*SE_AUC
    if lower < 0:
        lower = 0
    if upper > 1:
        upper = 1
    return (lower, upper)

def rsquareCI(y_true, y_pred, X, CI):
    '''
    adapted from: https://agleontyev.netlify.app/post/2019-09-05-calculating-r-squared-confidence-intervals/
    '''
    R2 = r2_score(y_true, y_pred)
    n = len(y_true)
    k = len(X.columns)
    SE = sqrt((4*R2*((1 - R2)**2)*((n - k - 1)**2))/((n**2 - 1)*(n + 3)))
    if CI == 0.67:
        upper = R2 + SE
        lower = R2 - SE
    elif CI == 0.8:
        upper = R2 + 1.3*SE
        lower = R2 - 1.3*SE
    elif CI == 0.95:
        upper = R2 + 2*SE
        lower = R2 - 2*SE
    elif CI == 0.99:
        upper = R2 + 2.6*SE
        lower = R2 - 2.6*SE
    else:
        raise ValueError('Unknown value for CI. Please use 0.67, 0.8, 0.95 or 0.99')
    return (lower, upper)

# make df of lightgbm classification metrics
def model_eval_metrics(preds, predictions, y_test):
    """ 
    Generate model accuracy metrics
    # """
    # # predicted values
    # predictions = model.predict(X_test)
    # # predicted probabilities
    # preds = model.predict_proba(X_test)[:, 1]
    # get precision and recall
    precision, recall, thresholds = precision_recall_curve(y_test, preds)
    # accuracy score
    accuracy = accuracy_score(y_test, predictions)
    # precision score
    p_score = precision_score(y_test, predictions)
    # recall score
    r_score = recall_score(y_test, predictions)
    # AUC of precision recall curve
    auc_prc = auc(recall, precision)
    # AUC of ROC
    auc_roc = roc_auc_score(y_test, preds)
    # AUC 95% CI
    ci = list(roc_auc_ci(y_test, preds))
    # partial AUC
    partial_auc = roc_auc_score(y_test, preds, max_fpr=0.05)
    # F1 score
    f1 = f1_score(y_test, predictions)
    
    # create df    
    df = pd.DataFrame([[p_score, r_score, accuracy, auc_prc, auc_roc, ci[0], ci[1], partial_auc, f1]])
    # set colnames
    df.columns = ['Precision', 'Recall', 'Accuracy', 'PR AUC', 'ROC AUC', 'AUC 2.5% CI', 'AUC 97.5% CI', 'Partial AUC', 'F1 score']
    
    return df

# F1-score evaluation metric function
def f1_score_lgbm(y_true, y_pred):
    y_pred = np.round(y_pred)
    f_score = f1_score(y_true, y_pred)
    is_higher_better = True
    return 'f1_score', f_score, is_higher_better

# Precision evaluation metric function
def precision_lgbm(y_true, y_pred):
    y_pred = np.round(y_pred)
    precision = precision_score(y_true, y_pred)
    is_higher_better = True
    return 'precision', precision, is_higher_better

# Precision-recall AUC evaluation metric function
def pr_auc_lgbm(y_true, y_pred):
    y_pred = np.round(y_pred)
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    auc_prc = auc(recall, precision)
    is_higher_better = True
    return 'pr_auc', auc_prc, is_higher_better

# Partial AUC evaluation metric function
def partial_auc_lgbm(y_true, y_pred):
    y_pred = np.round(y_pred)
    partial_auc = roc_auc_score(y_true, y_pred, max_fpr=0.05)
    is_higher_better = True
    return 'partial_auc', partial_auc, is_higher_better

# R squared evaluation metric function
def r2_score_lgbm(y_true, y_pred):
    if isinstance(y_pred, xgb.DMatrix):
        y_pred = y_pred.get_label()
    # y_pred = np.round(y_pred)
    r2 = r2_score(y_true, y_pred)
    is_higher_better = True
    return 'r2_score', r2, is_higher_better

def r2_score_xgb(y_true, y_pred):
    if isinstance(y_pred, xgb.DMatrix):
        y_pred = y_pred.get_label()
    r2 = r2_score(y_true, y_pred)
    return 'r2_score', r2


def find_best_trial(data):
    
    values = [trial.values for trial in data.trials]
    
    # Sort the data based on the second entry (index 1) in descending order
    sorted_data = sorted(values, key=lambda x: x[1], reverse=True)

    # Calculate the threshold for the top 5% of the second entry values
    threshold_index = int(len(sorted_data) * 0.05)
    threshold_value = sorted_data[threshold_index][1]

    # Subset study values to those with second entry above threshold
    subset_data = [study for study in sorted_data if study[1] >= threshold_value]

    # Find maximum AUC (first entry)
    best_values = max(subset_data, key=lambda t: t[0])
    
    # Get trial number with best values
    trial_number = [trial.number for trial in data.trials if trial.values == best_values][0]

    # Return best trial
    best_trial = data.trials[trial_number]
 
    return best_trial

def _check_classifer_response_method(estimator, response_method):
    if response_method not in ("predict_proba", "decision_function", "auto"):
        raise ValueError("response_method must be 'predict_proba', "
                         "'decision_function' or 'auto'")

    error_msg = "response method {} is not defined in {}"
    if response_method != "auto":
        prediction_method = getattr(estimator, response_method, None)
        if prediction_method is None:
            raise ValueError(error_msg.format(response_method,
                                              estimator.__class__.__name__))
    else:
        predict_proba = getattr(estimator, 'predict_proba', None)
        decision_function = getattr(estimator, 'decision_function', None)
        prediction_method = predict_proba or decision_function
        if prediction_method is None:
            raise ValueError(error_msg.format(
                "decision_function or predict_proba",
                estimator.__class__.__name__))

    return prediction_method

class RocCurveDisplay:
    '''
    ROC Curve visualization
    '''
    def __init__(self, fpr, tpr, roc_auc):
        self.fpr = fpr
        self.tpr = tpr
        self.roc_auc = roc_auc

    def plot(self, ax=None, name=None, **kwargs):
        '''
        Plot visualization
        
        Parameters
        ----------
        ax : matplotlib axes, default=None
            Axes object to plot on. If `None`, a new figure and axes is
            created.
        name : str, default=None
            Name of ROC Curve for labeling. If `None`, use the name of the
            estimator.
            
        Returns
        ----------
        display : :class:`~sklearn.metrics.plot.RocCurveDisplay`
            Object that stores computed values.
        '''
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots()

        name = self.estimator_name if name is None else name

        line_kwargs = {
            'label': "{} (AUC = {:0.3f})".format(name, self.roc_auc)
        }
        line_kwargs.update(**kwargs)

        self.line_ = ax.plot(self.fpr, self.tpr, **line_kwargs)[0]
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(loc='lower right')

        self.ax_ = ax
        self.figure_ = ax.figure
        return self


# plot an ROC curve
def plot_roc_curve(preds, y, sample_weight=None,
                   drop_intermediate=True, response_method="auto",
                   name=None, ax=None, **kwargs):
    '''
    Plot Receiver operating characteristic (ROC) curve
    
    Parameters
    ----------
    clf: trained classification model
    X,y: traning dataset
    name : Name of ROC Curve for labeling. If `None`, use the name of the estimator.
    
    Returns
    ----------
    viz: object storing computed fpr,tpr,auc,clf name
    '''

    # classification_error = ("{} should be a binary classifer".format(
    #     clf.__class__.__name__))
    # if not is_classifier(clf):
    #     raise ValueError(classification_error)

    # prediction_method = _check_classifer_response_method(clf, response_method)
    y_pred = preds
    # print(prediction_method)
    # y_pred = clf.predict_proba(X)
    # if y_pred.ndim != 1:
    #     if y_pred.shape[1] != 2:
    #         raise ValueError(classification_error)
    #     else:
    #         y_pred = y_pred[:, 1]

    pos_label = 1
    fpr, tpr, _ = roc_curve(
        y, 
        y_pred, 
        pos_label=pos_label,
        sample_weight=sample_weight,
        drop_intermediate=drop_intermediate
    )

    roc_auc = auc(fpr, tpr)
    viz = RocCurveDisplay(fpr, tpr, roc_auc)
    return viz.plot(ax=ax, name=name, **kwargs)

# perform LightGBM cross-validation and return dataframe of predicted values for all samples
def lgbm_cv_regression(
        nfolds:int, 
        random_seed:int, 
        data_dict:dict, 
        params:dict, 
        categorical_features:list = None,
        weights = None
    ):

    '''
    Test a LightGBM classification model using n-fold cross-validation and save an ROC curve plot
    
    Parameters
    ----------
    params: dictionary of model parameters.
    data_dict: dictionary with data. Must have elements 'X_train' and 'y_train'.
    nfolds: number of folds for cross-validation.
    random_seed: random seed.
    path: path for saving.
    date: date to add to ROC file name.
    title: title of ROC plot.
    filename: filename of ROC plot.
    image_type: type of image file to generate (e.g., "png", "pdf").
    
    Returns
    ----------
    auc_mean: mean AUC of model across the folds.
    '''

    # initialize stratified fold split
    folds = KFold(
        n_splits=nfolds, 
        shuffle=True,
        random_state=random_seed
    )

    # initialize list to document metrics across folds
    pred_list = []
    scores = []

    # get participant IDs
    all_ids = data_dict['X_train'].index

    # split train data into folds, fit model within each fold
    for i, (train_idxs, val_idxs) in enumerate(folds.split(data_dict['X_train'], data_dict['y_train'])):
        eval_data = {
            'X_val':data_dict['X_train'].iloc[val_idxs], 
            'y_val':data_dict['y_train'].iloc[val_idxs],
            'X_train':data_dict['X_train'].iloc[train_idxs], 
            'y_train':data_dict['y_train'].iloc[train_idxs]
        }
        
        if weights is not None: 
            sample_weight = weights[train_idxs]
            eval_sample_weight = [weights[val_idxs]]
                
        if weights is None: 
            sample_weight = None
            eval_sample_weight = None

        # initialize model
        model = lgb.LGBMRegressor(**params)

        # set callback
        if params['boosting_type'] == 'gbdt':
            callbacks = [lgb.log_evaluation(period=100), lgb.early_stopping(20, verbose=False)]
        elif params['boosting_type'] == 'dart':
            callbacks = None

        # set categorical features, if any
        if categorical_features is not None:
            # fit model in train data
            model.fit(
                X=eval_data['X_train'],
                y=eval_data['y_train'],
                eval_set=(eval_data['X_val'], eval_data['y_val']),
                eval_metric=r2_score_lgbm,
                categorical_feature=categorical_features,
                callbacks=callbacks,
                sample_weight=sample_weight,
                eval_sample_weight=eval_sample_weight
            )
        else:
            # fit model in train data
            model.fit(
                X=eval_data['X_train'],
                y=eval_data['y_train'],
                eval_set=(eval_data['X_val'], eval_data['y_val']),
                eval_metric=r2_score_lgbm,
                callbacks=callbacks,
                sample_weight=sample_weight,
                eval_sample_weight=eval_sample_weight
            )

        # get predicted values
        preds = model.predict(eval_data['X_val'])
        predicted_scores = model.predict(eval_data['X_val'], raw_score=True)

        # get R squared in the fold
        r2 = r2_score(eval_data['y_val'], preds)

        # append to list of R squared across folds
        scores.append(r2)

        # get participant IDs from data index
        ids = all_ids[val_idxs]

        # Create a dataframe for the fold with predicted values and participant IDs
        df = pd.DataFrame(index=ids)
        df['predicted_values'] = preds
        df['predicted_scores'] = predicted_scores
        
        # Append the fold dataframe to the list
        pred_list.append(df)

    # get average R squared across folds
    mean_r2 = np.mean(scores)

    # concatenate list of predictions across folds
    preds_df = pd.concat(pred_list)

    return mean_r2, preds_df

# perform LightGBM cross-validation and plot the ROC curves from each fold
def lgbm_cv_ROC(
        nfolds:int, 
        random_seed:int, 
        data_dict:dict, 
        params:dict, 
        path:str = None, 
        analysis:str = None,
        date:str = None, 
        title:str = None, 
        filename:str = None, 
        image_type:str = None, 
        plot:bool = True,
        metric_save:bool = True,
        scoring_metric:str = None, 
        multi_opt:bool = False,
        score1:str = None,
        score2:str = None,
        unbalanced:bool = False, 
        categorical_features:list = None,
        ensemble:bool = False,
        ensemble_ratio:float = 1,
        max_fpr:float = 0.05,
        weights = None
    ):

    '''
    Test a LightGBM classification model using n-fold cross-validation and save an ROC curve plot
    
    Parameters
    ----------
    params: dictionary of model parameters.
    data_dict: dictionary with data. Must have elements 'X_train' and 'y_train'.
    nfolds: number of folds for cross-validation.
    random_seed: random seed.
    path: path for saving.
    date: date to add to ROC file name.
    title: title of ROC plot.
    filename: filename of ROC plot.
    image_type: type of image file to generate (e.g., "png", "pdf").
    
    Returns
    ----------
    auc_mean: mean AUC of model across the folds.
    '''

    # initialize stratified fold split
    folds = StratifiedKFold(
        n_splits=nfolds, 
        shuffle=True,
        random_state=random_seed
    )

    # dictionary of scoring metric functions
    evals = {
        'f1_score': f1_score_lgbm,
        'pr_auc': pr_auc_lgbm,
        'precision': precision_lgbm,
        'partial_auc': partial_auc_lgbm,
        'r2_score': r2_score_lgbm,
        'auc': 'auc'
    }
    
    # set class weights
    if unbalanced:
        params.update({'is_unbalance': 'True'})

    # if other scoring_metric specified, add to dict
    if (scoring_metric is not None) & (scoring_metric not in list(evals.keys())):
        evals.update({scoring_metric: scoring_metric})

    # make list of metrics if multiple optimzation metrics
    if multi_opt:
        metric = [evals[score1], evals[score2]]
    else:
        metric = evals[scoring_metric]

    # initialize list to document metrics across folds
    df_list = []
    auc_scores = []
    scores = []
    aucs = []
    tprs = []
    pred_list = []
    mean_fpr = np.linspace(0, 1, 100)

    # get participant IDs
    all_ids = data_dict['X_train'].index

    if plot:
        # initialize ROC curve plot
        fig, ax = plt.subplots(figsize=(9, 6))
    
    # split train data into folds, fit model within each fold
    for i, (train_idxs, val_idxs) in enumerate(folds.split(data_dict['X_train'], data_dict['y_train'])):
        eval_data = {
            'X_test':data_dict['X_train'].iloc[val_idxs], 
            'y_test':data_dict['y_train'].iloc[val_idxs],
            'X_train':data_dict['X_train'].iloc[train_idxs], 
            'y_train':data_dict['y_train'].iloc[train_idxs]
        }
        
        if weights is not None: 
            sample_weight = weights[train_idxs]
            eval_sample_weight = [weights[val_idxs]]
            
        if weights is None: 
            sample_weight = None
            eval_sample_weight = None
        
        # Ensemble learning
        if ensemble:
            results = ensemble_learn(
                data_dict=eval_data,
                params=params,
                ratio=ensemble_ratio,
                scoring_metric=scoring_metric,
                multi_opt=multi_opt,
                score1=score1,
                score2=score2,
                max_fpr=max_fpr,
                unbalanced=unbalanced,
                sample_weight=sample_weight,
                eval_sample_weight=eval_sample_weight
            )
            
            # get predicted values
            preds = results['predicted_probabilities']
            predictions = results['predictions']
            predicted_scores = results['raw_scores']

            # get AUC for the fold
            fold_auc = results['roc_auc']
            # add AUC from fold to list of AUCs across folds
            auc_scores.append(fold_auc)
            
            # get classification model metrics
            if metric_save:
                # metric_df = model_eval_metrics(eval_data['X_test'], eval_data['y_test'], model)
                metric_df = model_eval_metrics(preds, predictions, eval_data['y_test'])
                df_list.append(metric_df)

            # get participant IDs from data index
            ids = all_ids[val_idxs]

            # Create a dataframe for the fold with predicted values and participant IDs
            df = pd.DataFrame(index=ids)
            df['predicted_probabilities'] = preds
            df['predicted_values'] = predictions
            df['predicted_scores'] = predicted_scores
            
            # Append the fold dataframe to the list
            pred_list.append(df)
            
            if plot:
                # ROC plot
                viz = plot_roc_curve(
                    preds,
                    eval_data['y_test'],
                    name=f'Fold {i+1}',
                    alpha=0.3, 
                    lw=1, 
                    ax=ax
                )

                # get true positive rate in the fold
                interp_tpr = interp(mean_fpr, viz.fpr, viz.tpr)
                interp_tpr[0] = 0.0
                # add to list of true positive rates across folds
                tprs.append(interp_tpr)
                # add AUC to list across folds
                aucs.append(viz.roc_auc)
                # get model score
                # scores.append(model.score(eval_data['X_val'], eval_data['y_val']))
        
        # Non-ensemble learning
        else:

            # initialize model
            model = lgb.LGBMClassifier(**params)

            # callbacks (early stopping)
            if params['boosting_type'] == 'gbdt':
                callbacks = [
                    # lgb.log_evaluation(period=100), 
                    lgb.early_stopping(20, verbose=False)
                ]
            elif params['boosting_type'] == 'dart':
                callbacks = None

            # set categorical features, if any
            if categorical_features is not None:
                # fit model in train data
                model.fit(
                    X=eval_data['X_train'],
                    y=eval_data['y_train'],
                    eval_set=(eval_data['X_test'], eval_data['y_test']),
                    eval_metric=metric,
                    categorical_feature=categorical_features,
                    callbacks=callbacks,
                    sample_weight=sample_weight,
                    eval_sample_weight=eval_sample_weight
                )
            else:
                # fit model in train data
                model.fit(
                    X=eval_data['X_train'],
                    y=eval_data['y_train'],
                    eval_set=(eval_data['X_test'], eval_data['y_test']),
                    eval_metric=metric,
                    callbacks=callbacks,
                    sample_weight=sample_weight,
                    eval_sample_weight=eval_sample_weight
                )

            # get predicted values
            preds = model.predict_proba(eval_data['X_test'])[:, 1]
            predictions = model.predict(eval_data['X_test'])
            predicted_scores = model.predict(eval_data['X_test'], raw_score=True)

            # get AUC for the fold
            fold_auc = roc_auc_score(eval_data['y_test'], preds)
            # add AUC from fold to list of AUCs across folds
            auc_scores.append(fold_auc)
            
            # get classification model metrics
            if metric_save:
                # metric_df = model_eval_metrics(eval_data['X_val'], eval_data['y_val'], model)
                metric_df = model_eval_metrics(preds, predictions, eval_data['y_test'])
                df_list.append(metric_df)

            # get participant IDs from data index
            ids = all_ids[val_idxs]

            # Create a dataframe for the fold with predicted values and participant IDs
            df = pd.DataFrame(index=ids)
            df['predicted_probabilities'] = preds
            df['predicted_values'] = predictions
            df['predicted_scores'] = predicted_scores
            
            # Append the fold dataframe to the list
            pred_list.append(df)
            
            if plot:
                # ROC plot
                viz = plot_roc_curve(
                    preds,
                    eval_data['y_test'],
                    name=f'Fold {i+1}',
                    alpha=0.3, 
                    lw=1, 
                    ax=ax
                )

                # get true positive rate in the fold
                interp_tpr = interp(mean_fpr, viz.fpr, viz.tpr)
                interp_tpr[0] = 0.0
                # add to list of true positive rates across folds
                tprs.append(interp_tpr)
                # add AUC to list across folds
                aucs.append(viz.roc_auc)
                # get model score
                # scores.append(model.score(eval_data['X_val'], eval_data['y_val']))

    if metric_save:
        # concatenate list of model metric dfs
        metric_df = pd.concat(df_list, axis=0, ignore_index=True)
        # set index names to fold names
        suffix = list(range(1, nfolds+1))
        suffix = [str(i) for i in suffix]
        prefix = list(np.repeat('Fold', nfolds, axis=0))
        index_names = [f'{i} {j}' for i, j in zip(prefix, suffix)]
        metric_df.index = index_names
        # add row with means across folds
        metric_df.loc['mean'] = df.mean()
        
        # save 
        name = f'{path}{analysis}_model_metrics_{date}.csv'
        metric_df.to_csv(
            name,
            header=True,
            index=True
        )

    # get average auc across folds
    auc_mean = np.mean(auc_scores)

    # concatenate list of predictions across folds
    preds_df = pd.concat(pred_list)

    ### plot
    if plot:
        ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='#b2182b',
                    label='Chance', alpha=.8)

        # take mean of true positive rates
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        # get mean auc as the mean false positive / mean true positive
        mean_auc = auc(mean_fpr, mean_tpr)
        # standard deviation of AUCs across the folds
        std_auc = np.std(aucs)

        # add mean false and true positive rates to plot
        ax.plot(
            mean_fpr, 
            mean_tpr, 
            color='#525252',
            label=r'Mean ROC (AUC = %0.3f $\pm$ %0.3f)' % (mean_auc, std_auc),
            lw=2, 
            alpha=0.8
        )

        # standard deviation of true positive rate
        std_tpr = np.std(tprs, axis=0)
        # confidence intervals of true positive rate
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)

        # add intervals to plot
        ax.fill_between(
            mean_fpr, 
            tprs_lower, 
            tprs_upper, 
            color='grey', 
            alpha=0.2,
            label=r'$\pm$ 1 std. dev.'
        )

        # set axis limits and title
        ax.set(
            xlim=[-0.05, 1.05], 
            ylim=[-0.05, 1.05],
            title=title
        )
        # legend
        ax.legend(loc="lower right", prop={'size': 10})

        # save
        name = f'{path}{filename}_{date}.{image_type}'
        plt.savefig(name)
        plt.close()

        print('Complete. Plot saved')

    return auc_mean, preds_df


# optuna hyperparameter optimization using cross-validation with a LightGBM classification model
def optuna_lgbm_cv(
        base_params:dict, 
        ntrials:int, 
        nfolds:int,
        random_seed:int, 
        data_dict:dict, 
        study_name:str, 
        storage:str, 
        scoring_metric:str = None, 
        unbalanced:bool = False, 
        categorical_features:list = None,
        multi_opt:bool = False,
        score1:str = None,
        score2:str = None,
        max_fpr:float = 0.05,
        ensemble:bool = False,
        ensemble_ratio:float = 1,
        weights = None,
        prune:bool = False,
        dart:bool = False
    ):

    '''
    Run Optuna hyperparameter tuning for a LightGBM model using n-fold cross-validation
    
    Parameters
    ----------
    base_params: model parameters for LightGBM that we aren't tuning.
    data_dict: dictionary with data. Must have elements 'X_train' and 'y_train'.
    nfolds: number of folds for cross-validation
    ntrials: number of trials for Optuna.
    random_seed: random seed. Any integer.
    study_name: name of study for initializing Optuna.
    storage: sql database path
    
    Returns
    ----------
    best_params: dictionary of best parameters selected by Optuna.
    '''
    
     # dictionary of eval metric functions 
    # these are not standard in lightgbm and need to be defined
    evals = {
        'f1_score': f1_score_lgbm,
        'pr_auc': pr_auc_lgbm,
        'precision': precision_lgbm,
        'partial_auc': partial_auc_lgbm,
        'r2_score': r2_score_lgbm,
        'auc': 'auc',
        'rmse': 'rmse'
    }
    
    # if other scoring_metric specified, add to dict
    if (scoring_metric is not None) & (scoring_metric not in list(evals.keys())):
        evals.update({scoring_metric: scoring_metric})
        
    if (score1 is not None) & (score1 not in list(evals.keys())):
        evals.update({score1: score1})
        
    if (score2 is not None) & (score2 not in list(evals.keys())):
        evals.update({score2: score2})
    
    # multiple or single scoring metric
    if multi_opt:
        metric = [evals[score1], evals[score2]]
    else:
        metric = evals[scoring_metric]

    # set method for splitting folds
    if base_params['objective'] == 'regression':
        split = KFold
    elif base_params['objective'] == 'binary':
        split = StratifiedKFold

    # initialize fold split
    folds = split(
        n_splits=nfolds, 
        shuffle=True,
        random_state=random_seed
    )

    # create optuna parameter search objective
    def objective(trial):
        
        # if we tune with dart
        if dart:
            #  parameter ranges to search
            if categorical_features is not None:
                tune_params = {
                    # booster type
                    'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart']),
                    # bagging fraction (fraction of data to be used for each iteration)
                    'subsample': trial.suggest_float('subsample', 0.1, 1.0),
                    # number of leaves per tree
                    'num_leaves': trial.suggest_int('num_leaves', 2, 256),
                    # max depth (# of levels in tree)
                    'max_depth': trial.suggest_int('max_depth', 1, 4),
                    # minimum number of data points in one leaf
                    'min_child_samples': trial.suggest_int('min_child_samples', 5, 300),
                    # learning rate (step size during gradient descent)
                    'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1, log=True),
                    # minimum number of samples required to be at a leaf node of the tree (similar to min_child_samples)
                    'min_child_weight': trial.suggest_float('min_child_weight', 1e-5, 100, log=True),
                    # L1 regularization
                    'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1, log=True),
                    # L2 regularization
                    'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1, log=True),
                    # sampling according to each tree.
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1.0),
                    "min_data_per_group": trial.suggest_int('min_data_per_group', 5, 300),
                    "cat_smooth": trial.suggest_float("cat_smooth", 1.0, 30.0),
                }
            else:
                tune_params = {
                    # booster type
                    'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart']),
                    # bagging fraction (fraction of data to be used for each iteration)
                    'subsample': trial.suggest_float('subsample', 0.1, 1.0),
                    # number of leaves per tree
                    'num_leaves': trial.suggest_int('num_leaves', 2, 256),
                    # max depth (# of levels in tree)
                    'max_depth': trial.suggest_int('max_depth', 1, 4),
                    # minimum number of data points in one leaf
                    'min_child_samples': trial.suggest_int('min_child_samples', 5, 300),
                    # learning rate (step size during gradient descent)
                    'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1, log=True),
                    # minimum number of samples required to be at a leaf node of the tree (similar to min_child_samples)
                    'min_child_weight': trial.suggest_float('min_child_weight', 1e-5, 100, log=True),
                    # L1 regularization
                    'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1, log=True),
                    # L2 regularization
                    'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1, log=True),
                    # sampling according to each tree.
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1.0)
                }
        
        # if we don't tune with dart
        else:
            if categorical_features is not None:
                tune_params = {
                    # bagging fraction (fraction of data to be used for each iteration)
                    'subsample': trial.suggest_float('subsample', 0.1, 1.0),
                    # number of leaves per tree
                    'num_leaves': trial.suggest_int('num_leaves', 2, 256),
                    # max depth (# of levels in tree)
                    # "max_depth": trial.suggest_int("max_depth", 3, 12),
                    # minimum number of data points in one leaf
                    'min_child_samples': trial.suggest_int('min_child_samples', 5, 300),
                    # learning rate (step size during gradient descent)
                    'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1, log=True),
                    # minimum number of samples required to be at a leaf node of the tree (similar to min_child_samples)
                    'min_child_weight': trial.suggest_float('min_child_weight', 1e-5, 100, log=True),
                    # L1 regularization
                    'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1, log=True),
                    # L2 regularization
                    'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1, log=True),
                    # sampling according to each tree.
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1.0),
                    "min_data_per_group": trial.suggest_int('min_data_per_group', 5, 300),
                    "cat_smooth": trial.suggest_float("cat_smooth", 1.0, 30.0),
                }
            else:
                tune_params = {
                    # bagging fraction (fraction of data to be used for each iteration)
                    'subsample': trial.suggest_float('subsample', 0.1, 1.0),
                    # number of leaves per tree
                    'num_leaves': trial.suggest_int('num_leaves', 2, 256),
                    # max depth (# of levels in tree)
                    # "max_depth": trial.suggest_int("max_depth", 3, 12),
                    # minimum number of data points in one leaf
                    'min_child_samples': trial.suggest_int('min_child_samples', 5, 300),
                    # learning rate (step size during gradient descent)
                    'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1, log=True),
                    # minimum number of samples required to be at a leaf node of the tree (similar to min_child_samples)
                    'min_child_weight': trial.suggest_float('min_child_weight', 1e-5, 100, log=True),
                    # L1 regularization
                    'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1, log=True),
                    # L2 regularization
                    'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1, log=True),
                    # sampling according to each tree.
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1.0)
                }
        
        # add in base params
        tune_params.update(base_params)
        
        # tune additional parameters for dart        
        if tune_params['boosting_type'] == 'dart':
            dart_params = {
                'drop_rate': trial.suggest_float('drop_rate', 0.1, 1.0),
                'skip_drop': trial.suggest_float('skip_drop',0.1, 1.0),
                'max_drop': trial.suggest_int('max_drop',1, 100)
            }
            tune_params.update(dart_params)
            tune_params.update({'drop_seed': random_seed})
                    
        # set metric param to 'None' for custom eval metrics
        if scoring_metric in ['f1_score', 'pr_auc', 'precision', "r2_score"]:
            tune_params.update({'metric': 'None'})

        # set class weights if dataset is unbalanced
        if unbalanced:
            tune_params.update({'is_unbalance': 'True'})

        # initialize list to document AUC across folds
        if multi_opt:
            scores1 = []
            scores2 = []
        else:
            scores = []

        # split training data into folds, fit model within each fold
        for i, (train_idxs, val_idxs) in enumerate(folds.split(data_dict['X_train'], data_dict['y_train'])):
            tune_data = {
                'X_test':data_dict['X_train'].iloc[val_idxs], 
                'y_test':data_dict['y_train'].iloc[val_idxs],
                'X_train':data_dict['X_train'].iloc[train_idxs], 
                'y_train':data_dict['y_train'].iloc[train_idxs]
            }
            
            if weights is not None: 
                sample_weight = weights[train_idxs]
                eval_sample_weight = [weights[val_idxs]]
                
            if weights is None: 
                sample_weight = None
                eval_sample_weight = None
            
            # Ensemble learning
            if ensemble:
                results = ensemble_learn(
                    data_dict=tune_data,
                    params=tune_params,
                    ratio=ensemble_ratio,
                    scoring_metric=scoring_metric,
                    multi_opt=multi_opt,
                    score1=score1,
                    score2=score2,
                    max_fpr=max_fpr,
                    unbalanced=unbalanced,
                    sample_weight=sample_weight,
                    eval_sample_weight=eval_sample_weight
                )
                
                if multi_opt:
                    scores1.append(results[score1])
                    scores2.append(results[score2])
                
                else:
                    scores.append(results[scoring_metric])
            
            # Non-ensemble learning
            else:    


                # set callbacks
                # if prune and i == 0:
                #     pruning_callback = optuna.integration.LightGBMPruningCallback(trial, scoring_metric)
                #     callbacks = [
                #         # lgb.log_evaluation(period=100), 
                #         lgb.early_stopping(20, verbose=False),
                #         pruning_callback
                #     ]
                # else:
                #     callbacks = [
                #         # lgb.log_evaluation(period=100), 
                #         lgb.early_stopping(20, verbose=False)
                #     ]
                        
                # set callbacks
                if tune_params['boosting_type'] == 'gbdt':
                    if prune and i == 0:
                        pruning_callback = optuna.integration.LightGBMPruningCallback(trial, scoring_metric)
                        callbacks = [
                            # lgb.log_evaluation(period=100), 
                            lgb.early_stopping(20, verbose=False),
                            pruning_callback
                        ]
                    else:
                        callbacks = [
                            # lgb.log_evaluation(period=100), 
                            lgb.early_stopping(20, verbose=False)
                        ]
                elif tune_params['boosting_type'] == 'dart':
                    if prune and i == 0:
                        callbacks = [optuna.integration.LightGBMPruningCallback(trial, scoring_metric)]
                    else:
                        callbacks = None
                                  
                # initialize model
                if base_params['objective'] == 'regression':
                    model = lgb.LGBMRegressor(**tune_params)
                else:
                    model = lgb.LGBMClassifier(**tune_params)
                        
                # set categorical features, if any
                if categorical_features is not None:
                    # fit model in train data
                    model.fit(
                        X=tune_data['X_train'],
                        y=tune_data['y_train'],
                        eval_set=(tune_data['X_test'], tune_data['y_test']),
                        eval_metric=metric,
                        categorical_feature=categorical_features,
                        callbacks=callbacks,
                        sample_weight=sample_weight,
                        eval_sample_weight=eval_sample_weight
                    )
                else:
                    # fit model in train data
                    model.fit(
                        X=tune_data['X_train'],
                        y=tune_data['y_train'],
                        eval_set=(tune_data['X_test'], tune_data['y_test']),
                        eval_metric=metric,
                        callbacks=callbacks,
                        sample_weight=sample_weight,
                        eval_sample_weight=eval_sample_weight
                    )
                
                # get predicted values from fold
                predictions = model.predict(tune_data['X_test'])

                if base_params['objective'] == 'binary':
                    preds = model.predict_proba(tune_data['X_test'])[:, 1]
                    # scoring metrics
                    f1 = f1_score(tune_data['y_test'], predictions)
                    precision, recall, thresholds = precision_recall_curve(tune_data['y_test'], preds)
                    pr_auc = auc(recall, precision)
                    partial_auc = roc_auc_score(tune_data['y_test'], preds, max_fpr=max_fpr)
                    roc_auc = roc_auc_score(tune_data['y_test'], preds)

                    metrics = {
                        'f1_score': f1,
                        'pr_auc': pr_auc,
                        'partial_auc': partial_auc,
                        'auc': roc_auc
                    }
                elif scoring_metric == 'r2_score':
                    # scoring metric
                    r2 = r2_score(tune_data['y_test'], predictions)
                    metrics = {
                        'r2_score': r2
                    }
                elif scoring_metric == 'rmse':
                    rmse = mean_squared_error(tune_data['y_test'], predictions, squared=False)
                    metrics = {
                        'rmse': rmse
                    }
                if multi_opt:
                    scores1.append(metrics[score1])
                    scores2.append(metrics[score2])
                
                else:
                    scores.append(metrics[scoring_metric])
            
        # return the average evaluation score over all folds
        if multi_opt:
            return np.mean(scores1), np.mean(scores2)
        else:
            return np.mean(scores)

    # set direction based on type of model
    if multi_opt:
        direction = ["maximize", "maximize"]
    elif scoring_metric == 'rmse':
        direction = "minimize"
    else:
        direction = "maximize"
    # elif base_params['objective'] == 'regression':
    #     direction = "minimize"
    # elif base_params['objective'] == 'binary':
    #     direction = "maximize"
    
    if multi_opt:   
        sampler = NSGAIISampler(seed=random_seed)
    else:
        sampler = TPESampler(seed=random_seed)

    # initialize optuna study
    if multi_opt:
        study = optuna.create_study(
            directions=direction,
            # Specify the storage URL.
            storage=storage,
            # storage=storage_name,    
            study_name=study_name,
            load_if_exists=False,
            sampler=sampler
        )
    else:
        if prune:
            study = optuna.create_study(
                direction=direction,
                # Specify the storage URL.
                storage=storage,
                study_name=study_name,
                load_if_exists=False,
                sampler=sampler,
                # pruner=optuna.pruners.MedianPruner(n_warmup_steps=50, n_startup_trials=5)
                pruner=optuna.pruners.ThresholdPruner(n_warmup_steps=50, lower=0.70)
            )
        else: 
            study = optuna.create_study(
                direction=direction,
                # Specify the storage URL.
                storage=storage,
                study_name=study_name,
                load_if_exists=True,
                sampler=sampler
            )

    # run optimization
    study.optimize(
        objective, 
        n_trials=ntrials, 
        gc_after_trial=True,
        timeout=None
        # n_jobs=4,
        # callbacks=[MaxTrialsCallback(ntrials, states=(TrialState.COMPLETE,))]
    )
    
    print("Tuning complete")

    # return optimized parameters
    return study


# optuna hyperparameter optimization using cross-validation with an XGBoost model
def optuna_xgb_cv(
        base_params:dict, 
        ntrials:int, 
        nfolds:int,
        random_seed:int, 
        data_dict:dict, 
        study_name:str, 
        storage:str, 
        scoring_metric:str = None, 
        unbalanced:bool = False, 
        categorical_features:list = None,
        multi_opt:bool = False,
        score1:str = None,
        score2:str = None,
        max_fpr:float = 0.05,
        ensemble:bool = False,
        ensemble_ratio:float = 1,
        weights = None,
        prune:bool = False
    ):

    '''
    Run Optuna hyperparameter tuning for a LightGBM model using n-fold cross-validation
    
    Parameters
    ----------
    base_params: model parameters for LightGBM that we aren't tuning.
    data_dict: dictionary with data. Must have elements 'X_train' and 'y_train'.
    nfolds: number of folds for cross-validation
    ntrials: number of trials for Optuna.
    random_seed: random seed. Any integer.
    study_name: name of study for initializing Optuna.
    storage: sql database path
    
    Returns
    ----------
    best_params: dictionary of best parameters selected by Optuna.
    '''
    
     # dictionary of eval metric functions 
    # these are not standard in xgb and need to be defined
    evals = {
        'f1_score': f1_score_lgbm,
        'pr_auc': pr_auc_lgbm,
        'precision': precision_lgbm,
        'partial_auc': partial_auc_lgbm,
        'r2_score': r2_score_xgb,
        'auc': 'auc'
    }
    
    # if other scoring_metric specified, add to dict
    if (scoring_metric is not None) & (scoring_metric not in list(evals.keys())):
        evals.update({scoring_metric: scoring_metric})
        
    if (score1 is not None) & (score1 not in list(evals.keys())):
        evals.update({score1: score1})
        
    if (score2 is not None) & (score2 not in list(evals.keys())):
        evals.update({score2: score2})
    
    # multiple or single scoring metric
    if multi_opt:
        metric = [evals[score1], evals[score2]]
    else:
        metric = evals[scoring_metric]

    # set method for splitting folds
    if 'reg' in base_params['objective']:
        split = KFold
    elif 'binary' in base_params['objective']:
        split = StratifiedKFold

    # initialize fold split
    folds = split(
        n_splits=nfolds, 
        shuffle=True,
        random_state=random_seed
    )

    # create optuna parameter search objective
    def objective(trial):
        
        # parameter ranges to search    
        tune_params = {
            'subsample': trial.suggest_float('subsample', 0.1, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 1.0),
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'min_child_weight': trial.suggest_float('min_child_weight', 1e-5, 100, log=True),
            'gamma': trial.suggest_float('gamma', 0, 1),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 50, 1000)
        }
        
        # add in base params
        tune_params.update(base_params)

        # initialize list to document AUC across folds
        if multi_opt:
            scores1 = []
            scores2 = []
        else:
            scores = []

        # split training data into folds, fit model within each fold
        for i, (train_idxs, val_idxs) in enumerate(folds.split(data_dict['X_train'], data_dict['y_train'])):
            tune_data = {
                'X_test':data_dict['X_train'].iloc[val_idxs], 
                'y_test':data_dict['y_train'].iloc[val_idxs],
                'X_train':data_dict['X_train'].iloc[train_idxs], 
                'y_train':data_dict['y_train'].iloc[train_idxs]
            }
            
            if weights is not None: 
                sample_weight = weights[train_idxs]
                eval_sample_weight = [weights[val_idxs]]
                
            if weights is None: 
                sample_weight = None
                eval_sample_weight = None
            
            # Ensemble learning
            if ensemble:
                results = ensemble_learn(
                    data_dict=tune_data,
                    params=tune_params,
                    ratio=ensemble_ratio,
                    scoring_metric=scoring_metric,
                    multi_opt=multi_opt,
                    score1=score1,
                    score2=score2,
                    max_fpr=max_fpr,
                    unbalanced=unbalanced,
                    sample_weight=sample_weight,
                    eval_sample_weight=eval_sample_weight
                )
                
                if multi_opt:
                    scores1.append(results[score1])
                    scores2.append(results[score2])
                
                else:
                    scores.append(results[scoring_metric])
            
            # Non-ensemble learning
            else:    
            
                # set metric param to 'None' for custom eval metrics
                # if scoring_metric in ['f1_score', 'pr_auc', 'precision', "r2_score"]:
                #     tune_params.update({'eval_metric': 'None'})
        
                # set class weights if dataset is unbalanced
                if unbalanced:
                    tune_params.update({'is_unbalance': 'True'})

                # set callbacks
                if prune and i == 0:
                    callbacks = [optuna.integration.XGBoostPruningCallback(trial, scoring_metric)]
                else:
                    callbacks = None
                    
                tune_params.update({'callbacks':callbacks})
                    
                # initialize model
                if 'reg' in base_params['objective']:
                    model = xgb.XGBRegressor(**tune_params)
                else:
                    model = xgb.XGBClassifier(**tune_params)
                    
                # set eval_metric
                # model.set_params(eval_metric=metric)
                    
                # set categorical features, if any
                if categorical_features is not None:
                    # fit model in train data
                    model.fit(
                        X=tune_data['X_train'],
                        y=tune_data['y_train'],
                        eval_set = [(tune_data['X_test'], tune_data['y_test'])],
                        # eval_metric=metric,
                        categorical_feature=categorical_features,
                        # callbacks=callbacks,
                        sample_weight=sample_weight,
                        sample_weight_eval_set=eval_sample_weight,
                        verbose=False
                    )
                else:
                    # fit model in train data
                    model.fit(
                        X=tune_data['X_train'],
                        y=tune_data['y_train'],
                        eval_set = [(tune_data['X_test'], tune_data['y_test'])],
                        # eval_metric=metric,
                        # callbacks=callbacks,
                        sample_weight=sample_weight,
                        sample_weight_eval_set=eval_sample_weight,
                        verbose=False
                    )
                
                # get predicted values from fold
                predictions = model.predict(tune_data['X_test'])

                if 'binary' in base_params['objective']:
                    preds = model.predict_proba(tune_data['X_test'])[:, 1]
                    # scoring metrics
                    f1 = f1_score(tune_data['y_test'], predictions)
                    precision, recall, thresholds = precision_recall_curve(tune_data['y_test'], preds)
                    pr_auc = auc(recall, precision)
                    partial_auc = roc_auc_score(tune_data['y_test'], preds, max_fpr=max_fpr)
                    roc_auc = roc_auc_score(tune_data['y_test'], preds)

                    metrics = {
                        'f1_score': f1,
                        'pr_auc': pr_auc,
                        'partial_auc': partial_auc,
                        'auc': roc_auc
                    }
                else:
                    # scoring metric
                    r2 = r2_score(tune_data['y_test'], predictions)
                    metrics = {
                        'r2_score': r2
                    }
                
                if multi_opt:
                    scores1.append(metrics[score1])
                    scores2.append(metrics[score2])
                
                else:
                    scores.append(metrics[scoring_metric])
                    # scores.append(r2)
            
        # return the average evaluation score over all folds
        if multi_opt:
            return np.mean(scores1), np.mean(scores2)
        else:
            return np.mean(scores)

    # set direction based on type of model
    if multi_opt:
        direction = ["maximize", "maximize"]
    else:
        direction = "maximize"
    # elif base_params['objective'] == 'regression':
    #     direction = "minimize"
    # elif base_params['objective'] == 'binary':
    #     direction = "maximize"
    
    if multi_opt:   
        sampler = NSGAIISampler(seed=random_seed)
    else:
        sampler = TPESampler(seed=random_seed)

    # initialize optuna study
    if multi_opt:
        study = optuna.create_study(
            directions=direction,
            # Specify the storage URL.
            storage=storage,
            # storage=storage_name,    
            study_name=study_name,
            load_if_exists=False,
            sampler=sampler
        )
    else:
        if prune:
            study = optuna.create_study(
                direction=direction,
                # Specify the storage URL.
                storage=storage,
                study_name=study_name,
                load_if_exists=False,
                sampler=sampler,
                # pruner=optuna.pruners.MedianPruner(n_warmup_steps=50, n_startup_trials=5)
                pruner=optuna.pruners.ThresholdPruner(n_warmup_steps=50, lower=0.70)
            )
        else: 
            study = optuna.create_study(
                direction=direction,
                # Specify the storage URL.
                storage=storage,
                study_name=study_name,
                load_if_exists=True,
                sampler=sampler
            )

    # run optimization
    study.optimize(
        objective, 
        n_trials=ntrials, 
        gc_after_trial=True,
        timeout=None
        # n_jobs=4,
        # callbacks=[MaxTrialsCallback(ntrials, states=(TrialState.COMPLETE,))]
    )
    
    print("Tuning complete")

    # return optimized parameters
    return study

# pipeline to run Boruta
def lgbm_pipeline_classification(
    base_params:dict, 
    ntrials:int, 
    nfolds:int, 
    random_seed:int, 
    data_dict:dict, 
    analysis:str,
    filepath:str, 
    perc:int,
    ROC_title_pre:str, 
    ROC_title_post:str,
    cm_title_pre:str,
    cm_title_post:str,
    image_type:str, 
    scoring_metric:str = None, 
    unbalanced:bool = False, 
    categorical_features:list = None,
    multi_opt:bool = False,
    score1:str = None,
    score2:str = None,
    max_fpr:float = 0.05,
    date:str = None,
    boruta:bool = False,
    boruta_trials:int = 100,
    ensemble:bool = False,
    ensemble_ratio:float = 1,
    sample_weight = None,
    eval_sample_weight = None
    ):

    '''
    Run Boruta feature selection process using LightGBM, Optuna, and n-fold cross-validation
    
    Parameters
    ----------
    base_params: model parameters for LightGBM that we aren't tuning.
    data_dict: dictionary with data. Must have elements 'X_train' and 'y_train'.
    nfolds: number of folds for cross-validation
    ntrials: number of trials for Optuna.
    random_seed: random seed. Any integer.
    study_name: name of study for initializing Optuna.
    storage: sql database path
    
    Returns
    ----------
    best_params: dictionary of best parameters selected by Optuna.
    '''
    
    ### initialization ----
    
    # fix error from deprecated numpy in boruta module
    np.bool = bool
    np.int = int
    np.float = float

    # set global random seed
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # date and time
    if date is None:
        now = dt.datetime.now()
        now = now.strftime('%Y-%m-%d_%H-%M-%S')
    else:
        now = date

    # dictionary of scoring metric functions
    evals = {
        'f1_score': f1_score_lgbm,
        'pr_auc': pr_auc_lgbm,
        'precision': precision_lgbm,
        'partial_auc': partial_auc_lgbm,
        'r2_score': r2_score_lgbm,
        'auc': 'auc'
    }

    target_names = {
        'f1_score': 'F1 score',
        'pr_auc': 'Precision-recall AUC',
        'precision': 'Precision',
        'partial_auc': 'Partial AUC',
        'r2_score': 'R squared',
        'auc': 'AUC'
    }

    # if other scoring_metric specified, add to dict
    if (scoring_metric is not None) & (scoring_metric not in list(evals.keys())):
        evals.update({scoring_metric: scoring_metric})

    # make list of metrics if multiple optimzation metrics
    if multi_opt:
        metric = [evals[score1], evals[score2]]
    else:
        metric = evals[scoring_metric]

    # set class weights
    if unbalanced:
        base_params.update({'is_unbalance': 'True'})

    # set categorical variables index
    if categorical_features is not None:
        cat_index = [data_dict['X_train'].columns.get_loc(c) for c in categorical_features]
    
    # create duplicates of train and test data before resampling to use after Boruta
    data_dict['X_train_post_boruta'] = data_dict['X_train']
    data_dict['X_test_post_boruta'] = data_dict['X_test']
    
    if boruta:
        
        # list of row indices at length of training data
        indices = list(range(len(data_dict['X_train'].index)))

        # create data dictionary for Boruta - further split training data into training and validation sets
        boruta_data = {}
        boruta_data['X_train'], boruta_data['X_val'], boruta_data['y_train'], boruta_data['y_val'], index_train, index_val = train_test_split(
            data_dict['X_train'], 
            data_dict['y_train'],
            indices,
            train_size=0.7,
            random_state=random_seed,
            stratify=data_dict['y_train']
        ) 
        
        if sample_weight is not None:
            sample_weight_boruta = sample_weight[index_train]
            eval_sample_weight_boruta = [sample_weight[index_val]]
            
        else:
            sample_weight_boruta = None
            eval_sample_weight_boruta = None
        
    ### Optuna hyperparameter optimization ----
    
    engine = create_engine(f'sqlite:///{filepath}data/{analysis}.db', echo=False)
    storage = f'sqlite:///{filepath}data/{analysis}.db'

    # set study name
    study_name = f'{analysis}_olink_params_{now}'
    
    # run optuna search and return the study
    study = optuna_lgbm_cv(
        base_params=base_params,
        nfolds=nfolds,
        ntrials=ntrials,
        random_seed=random_seed,
        data_dict=data_dict,
        multi_opt=multi_opt,
        scoring_metric=scoring_metric, 
        score1=score1,
        score2=score2,
        categorical_features=categorical_features,
        unbalanced=unbalanced,
        max_fpr=max_fpr,
        study_name=study_name,
        storage=storage,
        ensemble=ensemble,
        ensemble_ratio=ensemble_ratio,
        weights=sample_weight
    )
    
    if multi_opt:
        # get best trials
        print(f"Number of trials on the Pareto front (pre-Boruta): {len(study.best_trials)}", flush=True)
        best_trial = find_best_trial(study)
        print(f"Trial with highest accuracy: ", flush=True)
        print(f"\tnumber: {best_trial.number}", flush=True)
        print(f"\tparams: {best_trial.params}", flush=True)
        print(f"\tvalues: {best_trial.values}", flush=True)
                
        # save pareto front plot
        pio.renderers.default = "notebook"
        path = f'{filepath}output/optuna/{analysis}_optuna_pareto_plot_pre_Boruta_{now}.html'
        fig = optuna.visualization.plot_pareto_front(study, target_names=[target_names[score1], target_names[score2]])
        fig.write_html(path)
        
        # set best parameters
        best_params = best_trial.params
    else:
        best_params = study.best_params
    
    # save final params
    filename = f'{filepath}output/optuna/{analysis}_optuna_best_params_pre_Boruta_{now}.p'
    with open(filename, "wb") as f:
        pickle.dump(best_params, f, protocol=4)

    # save study
    filename = f'{filepath}output/optuna/{analysis}_optuna_study_pre_Boruta_{now}.p'
    with open(filename, "wb") as f:
        pickle.dump(study, f, protocol=4)


    ### Cross-validation pre-Boruta ----
    
    # update tuned parameters with base parameters
    best_params.update(base_params)
    
    # run cross-validation and save ROC plot, then return average AUC across folds
    cv_auc, pred_df_pre = lgbm_cv_ROC(
        params=best_params,
        nfolds=nfolds,
        random_seed=random_seed,
        data_dict=data_dict,
        unbalanced=unbalanced,
        multi_opt=multi_opt,
        scoring_metric=scoring_metric, 
        score1=score1,
        score2=score2,
        categorical_features=categorical_features,
        path=f'{filepath}output/single_model/',
        analysis=analysis,
        date=now,
        title=ROC_title_pre,
        filename=f'{analysis}_ROC_curves_pre_boruta',
        image_type=image_type,
        ensemble=ensemble,
        ensemble_ratio=ensemble_ratio,
        weights=sample_weight
    )


    ### Evaluation against test set pre-Boruta ----

    # Ensemble learning
    if ensemble:
        results = ensemble_learn(
            data_dict=data_dict,
            params=best_params,
            ratio=ensemble_ratio,
            scoring_metric=scoring_metric,
            multi_opt=multi_opt,
            score1=score1,
            score2=score2,
            max_fpr=max_fpr,
            unbalanced=unbalanced,
            sample_weight=sample_weight,
            eval_sample_weight=eval_sample_weight
        )
        
        # get predicted values
        preds = results['predicted_probabilities']
        predictions = results['predictions']
        eval_df = model_eval_metrics(preds, predictions, data_dict['y_test'])

    else:
        # Intialize model
        model = lgb.LGBMClassifier(**best_params)

        # callbacks (early stopping)
        if best_params['boosting_type'] == 'gbdt':
            callbacks = [
                # lgb.log_evaluation(period=100), 
                lgb.early_stopping(20, verbose=False)
            ]
        elif best_params['boosting_type'] == 'dart':
            callbacks = None

        # fit model
        if categorical_features is not None:
            model.fit(
                X=data_dict['X_train'], 
                y=data_dict['y_train'],  
                eval_set=[(data_dict['X_test'], data_dict['y_test'])],
                categorical_feature=categorical_features,
                eval_metric=metric,
                callbacks=callbacks,
                sample_weight=sample_weight,
                eval_sample_weight=eval_sample_weight
            )
        else:
            model.fit(
                X=data_dict['X_train'], 
                y=data_dict['y_train'],  
                eval_set=[(data_dict['X_test'], data_dict['y_test'])],
                eval_metric=metric,
                callbacks=callbacks,
                sample_weight=sample_weight,
                eval_sample_weight=eval_sample_weight
            )
    
        # evaluation metics
        preds = model.predict_proba(data_dict['X_test'])[:, 1]
        predictions = model.predict(data_dict['X_test'])
        eval_df = model_eval_metrics(preds, predictions, data_dict['y_test'])

        # save model
        filename = f'{filepath}results/single_model/{analysis}_pre_boruta_{now}.txt'
        model.booster_.save_model(
            filename,
            num_iteration=model.best_iteration_
        )
            
        filename = f'{filepath}results/single_model/{analysis}_pre_boruta_{now}.p'
        with open(filename, "wb") as f:
            pickle.dump(model, f, protocol=4)

    ### confusion matrix ----

    # figure name
    name = f'{filepath}output/single_model/{analysis}_confusion_matrix_pre_boruta_{now}.png'

    cm = confusion_matrix(
        data_dict['y_test'], 
        predictions, 
        normalize='pred'
    )
    color = 'white'
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, 
        display_labels=[0,1]
    )
    disp.plot(cmap=plt.cm.Blues)
    plt.title(cm_title_pre)
    plt.savefig(
        name,
        dpi=600,
        transparent=False,
        bbox_inches="tight"
    )
    plt.close()

    if boruta:
        
        ### Boruta ----
        
        print('Starting Boruta...', flush=True)

        # create an LGBM classifier object
        boruta_model = lgb.LGBMClassifier(**best_params)

        # initialize Boruta feature selection
        boruta = BoostBoruta(
            estimator=boruta_model, 
            max_iter=boruta_trials, 
            importance_type='shap_importances',
            perc=perc,
            n_jobs=-1,
            verbose=1
        )

        # callbacks (early stopping)
        if best_params['boosting_type'] == 'gbdt':
            callbacks = [
                # lgb.log_evaluation(period=100), 
                lgb.early_stopping(20, verbose=False)
            ]
        elif best_params['boosting_type'] == 'dart':
            callbacks = None

        ## try boruta and catch error if selects no features
        try:
            # fit Boruta model
            if categorical_features is not None:
                boruta.fit(
                    boruta_data['X_train'],
                    boruta_data['y_train'], 
                    eval_set=[(boruta_data['X_val'], boruta_data['y_val'])], 
                    eval_metric=metric,
                    categorical_feature=cat_index,
                    callbacks=callbacks,
                    sample_weight=sample_weight_boruta,
                    eval_sample_weight=eval_sample_weight_boruta
                )
            else:
                boruta.fit(
                    boruta_data['X_train'],
                    boruta_data['y_train'], 
                    eval_set=[(boruta_data['X_val'], boruta_data['y_val'])], 
                    eval_metric=metric,
                    callbacks=callbacks,
                    sample_weight=sample_weight_boruta,
                    eval_sample_weight=eval_sample_weight_boruta
                )

            # get selected features
            boruta_cols = boruta.support_
            selected_features = list(data_dict['X_train'].iloc[:, boruta_cols].columns)

            # print
            print("Features confirmed important by Boruta:", selected_features, flush=True)

            # subset categorical feature list to those selected by Boruta
            if categorical_features is not None:
                categorical_features = [var for var in categorical_features if var in selected_features]

            # create new X with only boruta-selected features
            data_dict.update({'X_train': data_dict['X_train_post_boruta'].iloc[:, boruta_cols]})
            data_dict.update({'X_test': data_dict['X_test_post_boruta'].iloc[:, boruta_cols]})

            # save Boruta model
            filename = f'{filepath}/results/boruta/{analysis}_Boruta_model_{now}.p'
            with open(filename, "wb") as f:
                pickle.dump(boruta, f, protocol=4)

            # save selected features 
            filename = f'{filepath}/results/boruta/{analysis}_Boruta_vars_to_keep_{now}.csv' 
            pd.Series(selected_features).to_csv(
                filename,
                header=False,
                index=False
            )
        
            ### Optuna hyperparameter optimization post-Boruta ----
            
            engine = create_engine(f'sqlite:///{filepath}data/{analysis}_post_boruta.db', echo=False)
            storage = f'sqlite:///{filepath}data/{analysis}_post_boruta.db'

            # set study name
            study_name = f'{analysis}_olink_params_post_boruta_{now}'

            # run optuna search and return the study
            study_post = optuna_lgbm_cv(
                base_params=base_params,
                nfolds=nfolds,
                ntrials=ntrials,
                random_seed=random_seed,
                data_dict=data_dict,
                multi_opt=multi_opt,
                scoring_metric=scoring_metric,
                score1=score1,
                score2=score2,
                categorical_features=categorical_features,
                unbalanced=unbalanced,
                max_fpr=max_fpr,
                study_name=study_name,
                storage=storage,
                ensemble=ensemble,
                ensemble_ratio=ensemble_ratio,
                weights=sample_weight
            )
            
            if multi_opt:
                # get best trials
                print(f"Number of trials on the Pareto front (post-Boruta): {len(study_post.best_trials)}", flush=True)
                best_trial = find_best_trial(study_post)
                print(f"Trial with highest accuracy: ", flush=True)
                print(f"\tnumber: {best_trial.number}", flush=True)
                print(f"\tparams: {best_trial.params}", flush=True)
                print(f"\tvalues: {best_trial.values}", flush=True)
                        
                # save pareto front plot
                pio.renderers.default = "notebook"
                path = f'{filepath}output/optuna/{analysis}_optuna_pareto_plot_post_Boruta_{now}.html'
                fig = optuna.visualization.plot_pareto_front(study_post, target_names=[target_names[score1], target_names[score2]])
                fig.write_html(path)
                
                # set best parameters
                best_params_post = best_trial.params
            else:
                best_params_post = study_post.best_params
            
            # save final params
            filename = f'{filepath}output/optuna/{analysis}_optuna_best_params_post_boruta_{now}.p'
            with open(filename, "wb") as f:
                pickle.dump(best_params_post, f, protocol=4)

            # save study
            filename = f'{filepath}output/optuna/{analysis}_optuna_study_post_Boruta_{now}.p'
            with open(filename, "wb") as f:
                pickle.dump(study_post, f, protocol=4)

            ### Cross-validation post-Boruta ----

            # update tuned parameters with base parameters
            best_params_post.update(base_params)
            
            # run cross-validation and save ROC plot, then return average AUC across folds
            cv_auc_post, pred_df_post = lgbm_cv_ROC(
                params=best_params_post,
                nfolds=nfolds,
                random_seed=random_seed,
                data_dict=data_dict,
                unbalanced=unbalanced,
                multi_opt=multi_opt,
                scoring_metric=scoring_metric, 
                score1=score1,
                score2=score2,
                categorical_features=categorical_features,
                path=f'{filepath}output/final_model/',
                analysis=analysis,
                date=now,
                title=ROC_title_post,
                filename=f'{analysis}_ROC_curves_post_boruta',
                image_type=image_type,
                ensemble=ensemble,
                ensemble_ratio=ensemble_ratio,
                weights=sample_weight
            )

            ### Evaluation against test set post-Boruta ----
            
            # class weights
            if unbalanced:
                best_params_post.update({'is_unbalance': 'True'})

            # Ensemble learning
            if ensemble:
                results = ensemble_learn(
                    data_dict=data_dict,
                    params=best_params_post,
                    ratio=ensemble_ratio,
                    scoring_metric=scoring_metric,
                    multi_opt=multi_opt,
                    score1=score1,
                    score2=score2,
                    max_fpr=max_fpr,
                    unbalanced=unbalanced,
                    sample_weight=sample_weight,
                    eval_sample_weight=eval_sample_weight
                )
                
                # get predicted values
                preds = results['predicted_probabilities']
                predictions = results['predictions']
                eval_df = model_eval_metrics(preds, predictions, data_dict['y_test'])

            else:
                # Intialize model
                model = lgb.LGBMClassifier(**best_params_post)

                # callbacks (early stopping)
                if best_params_post['boosting_type'] == 'gbdt':
                    callbacks = [
                        # lgb.log_evaluation(period=100), 
                        lgb.early_stopping(20, verbose=False)
                    ]
                elif best_params_post['boosting_type'] == 'dart':
                    callbacks = None

                # fit model
                if categorical_features is not None:
                    model.fit(
                        X=data_dict['X_train'], 
                        y=data_dict['y_train'],  
                        eval_set=[(data_dict['X_test'], data_dict['y_test'])],
                        categorical_feature=categorical_features,
                        eval_metric=metric,
                        callbacks=callbacks,
                        sample_weight=sample_weight,
                        eval_sample_weight=eval_sample_weight
                    )
                else:
                    model.fit(
                        X=data_dict['X_train'], 
                        y=data_dict['y_train'],  
                        eval_set=[(data_dict['X_test'], data_dict['y_test'])],
                        eval_metric=metric,
                        callbacks=callbacks,
                        sample_weight=sample_weight,
                        eval_sample_weight=eval_sample_weight
                    )
            
                # evaluation metics
                preds = model.predict_proba(data_dict['X_test'])[:, 1]
                predictions = model.predict(data_dict['X_test'])
                eval_df_post = model_eval_metrics(preds, predictions, data_dict['y_test'])
                
                # save model
                filename = f'{filepath}results/final_model/{analysis}_post_boruta_{now}.txt'
                model.booster_.save_model(
                    filename,
                    num_iteration=model.best_iteration_
                )
                
                filename = f'{filepath}results/final_model/{analysis}_post_boruta_{now}.p'
                with open(filename, "wb") as f:
                    pickle.dump(model, f, protocol=4)

            # Create a dataframe for the predicted values and participant IDs
            pred_df_test = pd.DataFrame(index=data_dict['X_test'].index)
            pred_df_test['predicted_probabilities'] = preds


            ### confusion matrix ----

            # figure name
            name = f'{filepath}output/final_model/{analysis}_confusion_matrix_post_boruta_{now}.png'

            cm = confusion_matrix(
                data_dict['y_test'], 
                predictions, 
                normalize='pred'
            )
            color = 'white'
            disp = ConfusionMatrixDisplay(
                confusion_matrix=cm, 
                display_labels=[0,1]
            )
            disp.plot(cmap=plt.cm.Blues)
            plt.title(cm_title_post)
            plt.savefig(
                name,
                dpi=600,
                transparent=False,
                bbox_inches="tight"
            )
            plt.close()
            
            # set flag that boruta did not fail
            boruta_catch = False
            
        # if it fails and selects no features, catch error and continue
        except RuntimeError as e:
            # Handle the error gracefully
            print("An error occurred:", str(e))
            print("Continuing with the rest of the script...")
            # set flag that boruta failed
            boruta_catch = True
            selected_features = 'None'
                
    ### SHAP values ----

    # get shap values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(data_dict['X_train'])

    if (boruta==True and boruta_catch==False):
        # get shap interactions
        shap_interactions = explainer.shap_interaction_values(data_dict['X_train'])

    # summary plot
    shap.summary_plot(
        shap_values[1], 
        data_dict['X_train'],
        max_display=25,
        show=False
    )

    # save summary plot
    name = f'{filepath}output/shap/plots/{analysis}_shap_summary_{now}.png'
    plt.savefig(
        name,
        dpi=600,
        transparent=False,
        bbox_inches="tight"
    )
    plt.close()

    # save shap values
    valsname = f'{filepath}output/shap/files/{analysis}_shap_values_{now}.p'
    with open(valsname, "wb") as f:
        pickle.dump(shap_values, f, protocol=4)

    if (boruta==True and boruta_catch==False):
        # save shap interaction values
        intvalsname = f'{filepath}output/shap/files/{analysis}_shap_interaction_values_{now}.p'
        with open(intvalsname, "wb") as f:
            pickle.dump(shap_interactions, f, protocol=4)
        
   
    #### save final accuracy metrics to text file ----
    sample_size = len(data_dict['X_train'].index)
    filename = f'{filepath}output/model_metrics/{analysis}_model_metrics_{now}.txt'

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    with open(filename, "a") as f:
        print("Training sample size:", sample_size, file=f)
        print("Cross-validation AUC (pre-Boruta):", cv_auc, file=f)
        print("Evaluation metrics in test data (pre-Boruta):", file=f)
        print(eval_df.to_string(index=False), file=f)
        if boruta:
            print(f"\nNumber of features confirmed important by Boruta: {len(selected_features)}", file=f)
            print("\nFeatures confirmed important by Boruta:", selected_features, file=f)
        if (boruta==True and boruta_catch==False):
            print("\nCross-validation AUC (post-Boruta):", cv_auc_post, file=f)
            print("Evaluation metrics in test data (post-Boruta):", file=f)
            print(eval_df_post.to_string(index=False), file=f)
            
    # make df to save
    test_auc_pre = eval_df['ROC AUC'].values[0]
    test_size = len(data_dict['X_test'].index)
    train_events = data_dict['y_train'].value_counts()[1]
    test_events = data_dict['y_test'].value_counts()[1]
    total_sample = sample_size + test_size
    total_events = train_events + test_events
    
    if boruta: 
        if boruta_catch==False:
        
            no_boruta_features = len(list(selected_features))
            test_auc_post = eval_df_post['ROC AUC'].values[0]
            
            # create df    
            df = pd.DataFrame(
                [[cv_auc, 
                test_auc_pre, 
                no_boruta_features, 
                cv_auc_post, 
                test_auc_post, 
                sample_size, 
                test_size, 
                train_events, 
                test_events, 
                total_sample, 
                total_events]]
            )
            
            # set column names
            df.columns = [
                'CV_AUC_pre', 
                'test_AUC_pre', 
                'no_boruta_features', 
                'CV_AUC_post', 
                'test_AUC_post', 
                'train_sample', 
                'test_sample', 
                'train_events', 
                'test_events', 
                'total_sample', 
                "total_events"
            ]
            
            # make class object
            results = BorutaClassification(
                best_params_post, 
                selected_features, 
                df, 
                eval_df, 
                eval_df_post, 
                pred_df_post, 
                pred_df_test
            )
    
        elif boruta_catch==True:
            
            no_boruta_features = 0
            cv_auc_post = None
            test_auc_post = None
            best_params_post=best_params
            eval_df_post = None
            pred_df_post = None
            pred_df_test = None
            
            # create df    
            df = pd.DataFrame(
                [[cv_auc, 
                test_auc_pre, 
                no_boruta_features, 
                cv_auc_post, 
                test_auc_post, 
                sample_size, 
                test_size, 
                train_events, 
                test_events, 
                total_sample, 
                total_events]]
            )
            
            # set column names
            df.columns = [
                'CV_AUC_pre', 
                'test_AUC_pre', 
                'no_boruta_features', 
                'CV_AUC_post', 
                'test_AUC_post', 
                'train_sample', 
                'test_sample', 
                'train_events', 
                'test_events', 
                'total_sample', 
                "total_events"
            ]
            
            # make class object
            results = BorutaClassification(
                best_params_post, 
                selected_features, 
                df, 
                eval_df, 
                eval_df_post, 
                pred_df_post, 
                pred_df_test
            )
    
    else:
        
        # create df    
        df = pd.DataFrame(
            [[cv_auc, 
            test_auc_pre, 
            sample_size, 
            test_size, 
            train_events, 
            test_events, 
            total_sample, 
            total_events]]
        )
        
        # set column names
        df.columns = [
            'CV_AUC_pre', 
            'test_AUC_pre', 
            'train_sample', 
            'test_sample', 
            'train_events', 
            'test_events', 
            'total_sample', 
            "total_events"
        ]
        
        # make class object
        results = LGBMClassification(
            best_params_post, 
            df
        )
        
    
    print('Finished.', flush=True)

    return results


# pipeline to run Boruta
def lgbm_classification(
    base_params:dict, 
    ntrials:int, 
    nfolds:int, 
    random_seed:int, 
    data_dict:dict, 
    analysis:str,
    filepath:str, 
    ROC_title:str, 
    cm_title:str,
    image_type:str, 
    scoring_metric:str = None, 
    unbalanced:bool = False, 
    categorical_features:list = None,
    multi_opt:bool = False,
    score1:str = None,
    score2:str = None,
    max_fpr:float = 0.05,
    date:str = None,
    ensemble:bool = False,
    ensemble_ratio:bool = 1,
    ensemble_parallel:bool = False
    ):

    '''
    Run Boruta feature selection process using LightGBM, Optuna, and n-fold cross-validation
    
    Parameters
    ----------
    base_params: model parameters for LightGBM that we aren't tuning.
    data_dict: dictionary with data. Must have elements 'X_train' and 'y_train'.
    nfolds: number of folds for cross-validation
    ntrials: number of trials for Optuna.
    random_seed: random seed. Any integer.
    study_name: name of study for initializing Optuna.
    storage: sql database path
    
    Returns
    ----------
    best_params: dictionary of best parameters selected by Optuna.
    '''
    
    ### initialization ----
    
    # set global random seed
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # date and time
    if date is None:
        now = dt.datetime.now()
        now = now.strftime('%Y-%m-%d_%H-%M-%S')
    else: 
        now = date

    # dictionary of scoring metric functions
    evals = {
        'f1_score': f1_score_lgbm,
        'pr_auc': pr_auc_lgbm,
        'precision': precision_lgbm,
        'partial_auc': partial_auc_lgbm,
        'r2_score': r2_score_lgbm,
        'auc': 'auc'
    }

    target_names = {
        'f1_score': 'F1 score',
        'pr_auc': 'Precision-recall AUC',
        'precision': 'Precision',
        'partial_auc': 'Partial AUC',
        'r2_score': 'R squared',
        'auc': 'AUC'
    }

    # if other scoring_metric specified, add to dict
    if (scoring_metric is not None) & (scoring_metric not in list(evals.keys())):
        evals.update({scoring_metric: scoring_metric})

    # make list of metrics if multiple optimzation metrics
    if multi_opt:
        metric = [evals[score1], evals[score2]]
    else:
        metric = evals[scoring_metric]

    # set class weights
    if unbalanced:
        base_params.update({'is_unbalance': 'True'})

    ### Optuna hyperparameter optimization ----
    
    engine = create_engine(f'sqlite:///{filepath}data/{analysis}.db', echo=False)
    storage = f'sqlite:///{filepath}data/{analysis}.db'

    # set study name
    study_name = f'{analysis}_olink_params_{now}'
    
    # run optuna search and return the study
    study = optuna_lgbm_cv(
        base_params=base_params,
        nfolds=nfolds,
        ntrials=ntrials,
        random_seed=random_seed,
        data_dict=data_dict,
        multi_opt=multi_opt,
        scoring_metric=scoring_metric, 
        score1=score1,
        score2=score2,
        categorical_features=categorical_features,
        unbalanced=unbalanced,
        max_fpr=max_fpr,
        study_name=study_name,
        storage=storage,
        ensemble=ensemble,
        ensemble_ratio=ensemble_ratio,
        ensemble_parallel=ensemble_parallel
    )
    
    if multi_opt:
        # get best trials
        print(f"Number of trials on the Pareto front (pre-Boruta): {len(study.best_trials)}", flush=True)
        best_trial = find_best_trial(study)
        print(f"Trial with highest accuracy: ", flush=True)
        print(f"\tnumber: {best_trial.number}", flush=True)
        print(f"\tparams: {best_trial.params}", flush=True)
        print(f"\tvalues: {best_trial.values}", flush=True)
                
        # save pareto front plot
        pio.renderers.default = "notebook"
        path = f'{filepath}output/optuna/{analysis}_optuna_pareto_plot_pre_Boruta_{now}.html'
        fig = optuna.visualization.plot_pareto_front(study, target_names=[target_names[score1], target_names[score2]])
        fig.write_html(path)
        
        # set best parameters
        best_params = best_trial.params
    else:
        best_params = study.best_params
    
    # save final params
    filename = f'{filepath}output/optuna/{analysis}_optuna_best_params_pre_Boruta_{now}.p'
    with open(filename, "wb") as f:
        pickle.dump(best_params, f, protocol=4)

    # save study
    filename = f'{filepath}output/optuna/{analysis}_optuna_study_pre_Boruta_{now}.p'
    with open(filename, "wb") as f:
        pickle.dump(study, f, protocol=4)


    ### Cross-validation ----
    
    # update tuned parameters with base parameters
    best_params.update(base_params)
    
    # run cross-validation and save ROC plot, then return average AUC across folds
    cv_auc, pred_df_pre = lgbm_cv_ROC(
        params=best_params,
        nfolds=nfolds,
        random_seed=random_seed,
        data_dict=data_dict,
        unbalanced=unbalanced,
        multi_opt=multi_opt,
        scoring_metric=scoring_metric, 
        score1=score1,
        score2=score2,
        categorical_features=categorical_features,
        path=f'{filepath}output/single_model/',
        analysis=analysis,
        date=now,
        title=ROC_title,
        filename=f'{analysis}_ROC_curves',
        image_type=image_type,
        ensemble=ensemble,
        ensemble_ratio=ensemble_ratio,
        ensemble_parallel=ensemble_parallel
    )


    ### Evaluation against test set ----
    
    # Ensemble learning
    if ensemble:
        results = ensemble_learn(
            data_dict=data_dict,
            params=best_params,
            ratio=ensemble_ratio,
            scoring_metric=scoring_metric,
            multi_opt=multi_opt,
            score1=score1,
            score2=score2,
            max_fpr=max_fpr,
            unbalanced=unbalanced
        )
        
        # get predicted values
        preds = results['predicted_probabilities']
        predictions = results['predictions']
        eval_df = model_eval_metrics(preds, predictions, data_dict['y_test'])

    else:
        # Intialize model
        model = lgb.LGBMClassifier(**best_params)

        # callbacks (early stopping)
        if best_params['boosting_type'] == 'gbdt':
            callbacks = [
                # lgb.log_evaluation(period=100), 
                lgb.early_stopping(20, verbose=False)
            ]
        elif best_params['boosting_type'] == 'dart':
            callbacks = None

        # fit model
        if categorical_features is not None:
            model.fit(
                X=data_dict['X_train'], 
                y=data_dict['y_train'],  
                eval_set=[(data_dict['X_test'], data_dict['y_test'])],
                categorical_feature=categorical_features,
                eval_metric=metric,
                callbacks=callbacks
            )
        else:
            model.fit(
                X=data_dict['X_train'], 
                y=data_dict['y_train'],  
                eval_set=[(data_dict['X_test'], data_dict['y_test'])],
                eval_metric=metric,
                callbacks=callbacks
            )

        # evaluation metics
        preds = model.predict(data_dict['X_test'])[:, 1]
        predictions = model.predict(data_dict['X_test'])
        eval_df = model_eval_metrics(preds, predictions, data_dict['y_test'])
        # eval_df = model_eval_metrics(data_dict['X_test'], data_dict['y_test'], model)

        # save model
        filename = f'{filepath}results/single_model/{analysis}_{now}.txt'
        model.booster_.save_model(
            filename,
            num_iteration=model.best_iteration_
        )

    ### confusion matrix ----

    # figure name
    name = f'{filepath}output/single_model/{analysis}_confusion_matrix_{now}.png'

    cm = confusion_matrix(
        data_dict['y_test'], 
        predictions, 
        normalize='pred'
    )
    color = 'white'
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, 
        display_labels=[0,1]
    )
    disp.plot(cmap=plt.cm.Blues)
    plt.title(cm_title)
    plt.savefig(
        name,
        dpi=600,
        transparent=False,
        bbox_inches="tight"
    )
    plt.close()

    ### SHAP values ----

    # # get shap values
    # explainer = shap.TreeExplainer(model)
    # shap_values = explainer.shap_values(data_dict['X_train'])

    # # get shap interactions
    # # shap_interactions = explainer.shap_interaction_values(data_dict['X_train'])

    # # summary plot
    # shap.summary_plot(
    #     shap_values[1], 
    #     data_dict['X_train'],
    #     max_display=25,
    #     show=False
    # )

    # # save summary plot
    # name = f'{filepath}output/shap/plots/{analysis}_shap_summary_{now}.png'
    # plt.savefig(
    #     name,
    #     dpi=600,
    #     transparent=False,
    #     bbox_inches="tight"
    # )
    # plt.close()

    # # save shap values
    # valsname = f'{filepath}output/shap/files/{analysis}_shap_values_{now}.p'
    # with open(valsname, "wb") as f:
    #     pickle.dump(shap_values, f, protocol=4)

    # save shap interaction values
    # intvalsname = f'{filepath}output/shap/files/{analysis}_shap_interaction_values_{now}.p'
    # with open(intvalsname, "wb") as f:
    #     pickle.dump(shap_interactions, f, protocol=4)
        
   
    #### save final accuracy metrics to text file ----
    sample_size = len(data_dict['X_train'].index)
    filename = f'{filepath}output/model_metrics/{analysis}_model_metrics_{now}.txt'

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    with open(filename, "a") as f:
        print(f"Full sample size: {len(data_dict['X_train'].index) + len(data_dict['X_test'].index)}", file=f)
        print("Training sample size:", sample_size, file=f)
        print("Cross-validation AUC:", cv_auc, file=f)
        print("Evaluation metrics in test data:", file=f)
        print(eval_df.to_string(index=False), file=f)
        
    # make df to save
    test_size = len(data_dict['X_test'].index)
    train_events = data_dict['y_train'].value_counts()[1]
    test_events = data_dict['y_test'].value_counts()[1]
    total_sample = sample_size + test_size
    total_events = train_events + test_events
    
     # create df    
    df = pd.DataFrame(
        [[sample_size, 
          test_size, 
          train_events, 
          test_events, 
          total_sample, 
          total_events,
          cv_auc]]
    )
    
    # set column names
    df.columns = [
        'train_sample', 
        'test_sample', 
        'train_events', 
        'test_events', 
        'total_sample', 
        "total_events",
        'Train CV ROC AUC'
    ]
    
    # change column names
    colnames = list(eval_df.columns)
    colnames = [f'Test {col}' for col in colnames]
    eval_df.columns = colnames
    
    df = pd.concat([df, eval_df], axis=1)
    
    # make class object
    results = LGBMClassification(
        best_params, 
        df
    )
    
    print('Finished.', flush=True)

    return results

# pipeline to run Boruta
def lgbm_pipeline_regression(
    base_params:dict, 
    ntrials:int, 
    nfolds:int, 
    random_seed:int, 
    data_dict:dict, 
    analysis:str,
    filepath:str, 
    perc:int,
    scoring_metric:str = None, 
    unbalanced:bool = False, 
    categorical_features:list = None,
    multi_opt:bool = False,
    score1:str = None,
    score2:str = None,
    max_fpr:float = 0.05,
    date:str = None,
    boruta:bool = False,
    boruta_trials:int = 200,
    pre_boruta_optuna_params:str = None,
    dart:bool = False,
    sample_weight = None,
    eval_sample_weight = None
    ):

    '''
    Run Boruta feature selection process using LightGBM, Optuna, and n-fold cross-validation
    
    Parameters
    ----------
    base_params: model parameters for LightGBM that we aren't tuning.
    data_dict: dictionary with data. Must have elements 'X_train' and 'y_train'.
    nfolds: number of folds for cross-validation
    ntrials: number of trials for Optuna.
    random_seed: random seed. Any integer.
    study_name: name of study for initializing Optuna.
    storage: sql database path
    
    Returns
    ----------
    best_params: dictionary of best parameters selected by Optuna.
    '''
    
    ### initialization ----

    # fix error from deprecated numpy in boruta module
    np.bool = bool
    np.int = int
    np.float = float
    
    # set global random seed
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # date and time
    if date is None:
        now = dt.datetime.now()
        now = now.strftime('%Y-%m-%d')
    else:
        now = date

    # dictionary of scoring metric functions
    evals = {
        'r2_score': r2_score_lgbm
    }

    target_names = {
        'r2_score': 'R squared'
    }

    # if other scoring_metric specified, add to dict
    if (scoring_metric is not None) & (scoring_metric not in list(evals.keys())):
        evals.update({scoring_metric: scoring_metric})

    # make list of metrics if multiple optimzation metrics
    if multi_opt:
        metric = [evals[score1], evals[score2]]
    else:
        metric = evals[scoring_metric]

    # set categorical variables index
    if categorical_features is not None:
        cat_index = [data_dict['X_train'].columns.get_loc(c) for c in categorical_features]

    ### Optuna hyperparameter optimization ----
    
    # if no previous params to give, run optuna
    if pre_boruta_optuna_params is None:

        engine = create_engine(f'sqlite:///{filepath}data/{analysis}.db', echo=False)
        storage = f'sqlite:///{filepath}data/{analysis}.db'

        # set study name
        study_name = f'{analysis}_olink_params_{now}'
        
        # run optuna search and return the study
        study = optuna_lgbm_cv(
            base_params=base_params,
            nfolds=nfolds,
            ntrials=ntrials,
            random_seed=random_seed,
            data_dict=data_dict,
            scoring_metric=scoring_metric,
            multi_opt=multi_opt,
            score1=score1,
            score2=score2,
            categorical_features=categorical_features,
            unbalanced=unbalanced,
            max_fpr=max_fpr,
            study_name=study_name,
            storage=storage,
            weights=sample_weight,
            dart=dart
        )
        
        if multi_opt:
            # get best trials
            print(f"Number of trials on the Pareto front (pre-Boruta): {len(study.best_trials)}", flush=True)
            best_trial = find_best_trial(study)
            print(f"Trial with highest accuracy: ", flush=True)
            print(f"\tnumber: {best_trial.number}", flush=True)
            print(f"\tparams: {best_trial.params}", flush=True)
            print(f"\tvalues: {best_trial.values}", flush=True)
                    
            # save pareto front plot
            pio.renderers.default = "notebook"
            path = f'{filepath}output/optuna/{analysis}_optuna_pareto_plot_pre_Boruta_{now}.html'
            fig = optuna.visualization.plot_pareto_front(study, target_names=[target_names[score1], target_names[score2]])
            fig.write_html(path)
            
            # set best parameters
            best_params = best_trial.params
        else:
            best_params = study.best_params
        
        # save final params
        filename = f'{filepath}output/optuna/{analysis}_optuna_best_params_pre_Boruta_{now}.p'
        with open(filename, "wb") as f:
            pickle.dump(best_params, f, protocol=4)

        # save study
        filename = f'{filepath}output/optuna/{analysis}_optuna_study_pre_Boruta_{now}.p'
        with open(filename, "wb") as f:
            pickle.dump(study, f, protocol=4)

    else:
        # load previous params
        with open(pre_boruta_optuna_params, "rb") as f:
            best_params = pickle.load(f)

    ### Cross-validation in training data pre-Boruta ----
    
    # update tuned parameters with base parameters
    best_params.update(base_params)

    cv_r2_pre, pred_df_pre = lgbm_cv_regression(
        nfolds=nfolds, 
        random_seed=random_seed, 
        data_dict=data_dict, 
        params=best_params,
        weights=sample_weight
    )


    ### Evaluation against test set pre-Boruta ----

    # update tuned parameters with base parameters
    best_params.update(base_params)

    # Intialize model
    model = lgb.LGBMRegressor(**best_params)

    # callbacks (early stopping)
    if best_params['boosting_type'] == 'gbdt':
        callbacks = [
            # lgb.log_evaluation(period=100), 
            lgb.early_stopping(20, verbose=False)
        ]
    elif best_params['boosting_type'] == 'dart':
        callbacks = None

    # fit model
    if categorical_features is not None:
        model.fit(
            X=data_dict['X_train'], 
            y=data_dict['y_train'],  
            eval_set=[(data_dict['X_test'], data_dict['y_test'])],
            categorical_feature=categorical_features,
            eval_metric=metric,
            callbacks=callbacks,
            sample_weight=sample_weight,
            eval_sample_weight=eval_sample_weight
        )
    else:
        model.fit(
            X=data_dict['X_train'], 
            y=data_dict['y_train'],  
            eval_set=[(data_dict['X_test'], data_dict['y_test'])],
            eval_metric=metric,
            callbacks=callbacks,
            sample_weight=sample_weight,
            eval_sample_weight=eval_sample_weight
        )

    # evaluation metics
    predictions = model.predict(data_dict['X_test'])
    r2_pre = r2_score(data_dict['y_test'], predictions)
    r2_ci_pre = list(rsquareCI(data_dict['y_test'], predictions, data_dict['X_test'], 0.95))

    # save model as text
    filename = f'{filepath}results/single_model/{analysis}_pre_boruta_{now}.txt'
    model.booster_.save_model(
        filename,
        num_iteration=model.best_iteration_
    )
    
    # save model as pickle
    filename = f'{filepath}results/single_model/{analysis}_pre_boruta_{now}.p'
    with open(filename, "wb") as f:
        pickle.dump(model, f, protocol=4)

    if boruta:
        
        ### Boruta ----
        
        print('Starting Boruta...', flush=True)

        boruta_data = {}
        
        # list of row indices at length of training data
        indices = list(range(len(data_dict['X_train'].index)))

        # further split training data into training and validation sets
        boruta_data['X_train'], boruta_data['X_val'], boruta_data['y_train'], boruta_data['y_val'], index_train, index_val = train_test_split(
            data_dict['X_train'], 
            data_dict['y_train'],
            indices,
            train_size=0.7,
            random_state=random_seed
        ) 
        
        if sample_weight is not None:
            sample_weight_boruta = sample_weight[index_train]
            eval_sample_weight_boruta = [sample_weight[index_val]]
            
        else:
            sample_weight_boruta = None
            eval_sample_weight_boruta = None

        # create an LGBM classifier object
        model = lgb.LGBMRegressor(**best_params)

        # initialize Boruta feature selection
        boruta = BoostBoruta(
            estimator=model, 
            max_iter=boruta_trials, 
            importance_type='shap_importances',
            perc=perc,
            n_jobs=-1,
            verbose=1
        )

        # callbacks (early stopping)
        if best_params['boosting_type'] == 'gbdt':
            callbacks = [
                # lgb.log_evaluation(period=100), 
                lgb.early_stopping(20, verbose=False)
            ]
        elif best_params['boosting_type'] == 'dart':
            callbacks = None

        # fit Boruta model
        if categorical_features is not None:
            boruta.fit(
                boruta_data['X_train'],
                boruta_data['y_train'], 
                eval_set=[(boruta_data['X_val'], boruta_data['y_val'])], 
                eval_metric=metric,
                categorical_feature=cat_index,
                callbacks=callbacks,
                sample_weight=sample_weight_boruta,
                eval_sample_weight=eval_sample_weight_boruta
            )
        else:
            boruta.fit(
                boruta_data['X_train'],
                boruta_data['y_train'], 
                eval_set=[(boruta_data['X_val'], boruta_data['y_val'])], 
                eval_metric=metric,
                callbacks=callbacks,
                sample_weight=sample_weight_boruta,
                eval_sample_weight=eval_sample_weight_boruta
            )

        # get selected features
        boruta_cols = boruta.support_
        selected_features = list(data_dict['X_train'].iloc[:, boruta_cols].columns)

        # print
        print("Features confirmed important by Boruta:", selected_features, flush=True)

        # subset categorical feature list to those selected by Boruta
        if categorical_features is not None:
            categorical_features = [var for var in categorical_features if var in selected_features]

        # create new X with only boruta-selected features
        data_dict.update({'X_train': data_dict['X_train'].iloc[:, boruta_cols]})
        data_dict.update({'X_test': data_dict['X_test'].iloc[:, boruta_cols]})

        # save Boruta model
        filename = f'{filepath}/results/boruta/{analysis}_Boruta_model_{now}.p'
        with open(filename, "wb") as f:
            pickle.dump(boruta, f, protocol=4)

        # save selected features 
        filename = f'{filepath}/results/boruta/{analysis}_Boruta_vars_to_keep_{now}.csv' 
        pd.Series(selected_features).to_csv(
            filename,
            header=False,
            index=False
        )
        
        ### Optuna hyperparameter optimization post-Boruta ----
        
        engine = create_engine(f'sqlite:///{filepath}data/{analysis}_post_boruta.db', echo=False)
        storage = f'sqlite:///{filepath}data/{analysis}_post_boruta.db'

        # set study name
        study_name = f'{analysis}_olink_params_post_boruta_{now}'

        # run optuna search and return the study
        study_post = optuna_lgbm_cv(
            base_params=base_params,
            nfolds=nfolds,
            ntrials=ntrials,
            random_seed=random_seed,
            data_dict=data_dict,
            scoring_metric=scoring_metric,
            multi_opt=multi_opt,
            score1=score1,
            score2=score2,
            categorical_features=categorical_features,
            unbalanced=unbalanced,
            max_fpr=max_fpr,
            study_name=study_name,
            storage=storage,
            weights=sample_weight,
            dart=dart
        )
        
        if multi_opt:
            # get best trials
            print(f"Number of trials on the Pareto front (post-Boruta): {len(study_post.best_trials)}", flush=True)
            best_trial = find_best_trial(study_post)
            print(f"Trial with highest accuracy: ", flush=True)
            print(f"\tnumber: {best_trial.number}", flush=True)
            print(f"\tparams: {best_trial.params}", flush=True)
            print(f"\tvalues: {best_trial.values}", flush=True)
                    
            # save pareto front plot
            pio.renderers.default = "notebook"
            path = f'{filepath}output/optuna/{analysis}_optuna_pareto_plot_post_Boruta_{now}.html'
            fig = optuna.visualization.plot_pareto_front(study_post, target_names=[target_names[score1], target_names[score2]])
            fig.write_html(path)
            
            # set best parameters
            best_params_post = best_trial.params
        else:
            best_params_post = study_post.best_params
        
        # save final params
        filename = f'{filepath}output/optuna/{analysis}_optuna_best_params_post_boruta_{now}.p'
        with open(filename, "wb") as f:
            pickle.dump(best_params_post, f, protocol=4)

        # save study
        filename = f'{filepath}output/optuna/{analysis}_optuna_study_post_Boruta_{now}.p'
        with open(filename, "wb") as f:
            pickle.dump(study_post, f, protocol=4)

        ### Cross-validation in training data post-Boruta ----
        
        # update tuned parameters with base parameters
        best_params_post.update(base_params)

        cv_r2_post, pred_df_post = lgbm_cv_regression(
            nfolds=nfolds, 
            random_seed=random_seed, 
            data_dict=data_dict, 
            params=best_params_post,
            weights=sample_weight
        )

        ### Evaluation against test set post-Boruta ----
        
        # update tuned parameters with base parameters
        best_params_post.update(base_params)
        
        # Intialize model
        model = lgb.LGBMRegressor(**best_params_post)

        # callbacks (early stopping)
        if best_params_post['boosting_type'] == 'gbdt':
            callbacks = [
                # lgb.log_evaluation(period=100), 
                lgb.early_stopping(20, verbose=False)
            ]
        elif best_params_post['boosting_type'] == 'dart':
            callbacks = None

        # fit model
        if categorical_features is not None:
            model.fit(
                X=data_dict['X_train'], 
                y=data_dict['y_train'],  
                eval_set=[(data_dict['X_test'], data_dict['y_test'])],
                categorical_feature=categorical_features,
                eval_metric=metric,
                callbacks=callbacks,
                sample_weight=sample_weight,
                eval_sample_weight=eval_sample_weight
            )
        else:
            model.fit(
                X=data_dict['X_train'], 
                y=data_dict['y_train'],  
                eval_set=[(data_dict['X_test'], data_dict['y_test'])],
                eval_metric=metric,
                callbacks=callbacks,
                sample_weight=sample_weight,
                eval_sample_weight=eval_sample_weight
            )

        # evaluation metics
        predictions = model.predict(data_dict['X_test'])
        r2_post = r2_score(data_dict['y_test'], predictions)
        r2_ci_post = list(rsquareCI(data_dict['y_test'], predictions, data_dict['X_test'], 0.95))
 
        # save model as text
        filename = f'{filepath}results/final_model/{analysis}_post_boruta_{now}.txt'
        model.booster_.save_model(
            filename,
            num_iteration=model.best_iteration_
        )
        
        # save model as pickle
        filename = f'{filepath}results/final_model/{analysis}_post_boruta_{now}.p'
        with open(filename, "wb") as f:
            pickle.dump(model, f, protocol=4)


    ### SHAP values ----

    # get shap values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(data_dict['X_train'])

    # summary plot
    shap.summary_plot(
        shap_values, 
        data_dict['X_train'],
        max_display=25,
        show=False
    )

    # save summary plot
    name = f'{filepath}output/shap/plots/{analysis}_shap_summary_{now}.png'
    plt.savefig(
        name,
        dpi=600,
        transparent=False,
        bbox_inches="tight"
    )
    plt.close()
    
    # convert shap values to df and save index as eid
    shap_values = pd.DataFrame(shap_values, columns=list(data_dict['X_train'].columns))
    shap_values['eid'] = data_dict['X_train'].index
    shap_values.set_index('eid', inplace=True)

    # save shap values
    saveObject = (shap_values, data_dict['X_train'])
    valsname = f'{filepath}output/shap/files/{analysis}_shap_values_{now}.p'
    with open(valsname, "wb") as f:
        pickle.dump(saveObject, f, protocol=4)


    if boruta and sample_weight is None:
        # get shap interactions
        shap_interactions = explainer.shap_interaction_values(data_dict['X_train'])

        # save shap interaction values
        saveObject = (shap_interactions, data_dict['X_train'])
        intvalsname = f'{filepath}output/shap/files/{analysis}_shap_interaction_values_{now}.p'
        with open(intvalsname, "wb") as f:
            pickle.dump(saveObject, f, protocol=4)
            
   
    #### save final accuracy metrics to text file ----
    sample_size = len(data_dict['X_train'].index)
    filename = f'{filepath}output/model_metrics/{analysis}_model_metrics_{now}.txt'

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    with open(filename, "a") as f:
        print(f"Train sample size: {sample_size}", file=f)
        print(f"Avg. R squared from CV in train data (pre-Boruta): {cv_r2_pre:.4f}", file=f)
        # print(f"R squared in test data (pre-Boruta): {r2_pre:.4f}", file=f)
        print(f"R squared in test data (pre-Boruta): {r2_pre:.4f}, 95% CI: {r2_ci_pre[0]:.4f}-{r2_ci_pre[1]:.4f}", file=f)
        if boruta:
            print(f"\nNumber of features confirmed important by Boruta: {len(selected_features)}", file=f)
            print(f"\nFeatures confirmed important by Boruta: {selected_features}", file=f)
            print(f"\nAvg. R squared from CV in train data (post-Boruta): {cv_r2_post}", file=f)
            # print(f"R squared in test data (post-Boruta): {r2_post:.4f}", file=f)
            print(f"R squared in test data (post-Boruta): {r2_post:.4f}, 95% CI: {r2_ci_post[0]:.4f}-{r2_ci_post[1]:.4f}", file=f)

    print('Finished.', flush=True)

    if boruta:
        # make df to save
        no_boruta_features = len(list(selected_features))
        test_size = len(data_dict['X_test'].index)
        total_sample = sample_size + test_size
        
        # create df    
        df = pd.DataFrame(
            [[cv_r2_pre, 
            r2_pre, 
            no_boruta_features, 
            cv_r2_post, 
            r2_post, 
            sample_size, 
            test_size, 
            total_sample]]
        )
        
        # set column names
        df.columns = [
            'CV_R2_pre', 
            'test_R2_pre', 
            'no_boruta_features', 
            'CV_R2_post', 
            'test_R2_post', 
            'train_sample', 
            'test_sample', 
            'total_sample'
        ]
        
        # make class object
        results = BorutaRegression(
            best_params_post, 
            selected_features,
            df
        )
        
    else:
        test_size = len(data_dict['X_test'].index)
        total_sample = sample_size + test_size
        
        # create df    
        df = pd.DataFrame(
            [[cv_r2_pre, 
            r2_pre, 
            sample_size, 
            test_size, 
            total_sample]]
        )
        
        # set column names
        df.columns = [
            'CV_R2_pre', 
            'test_R2_pre', 
            'train_sample', 
            'test_sample', 
            'total_sample'
        ]
    
        # make class object
        results = LGBMRegression(
            best_params, 
            df
        )

    print('Finished.', flush=True)

    return results