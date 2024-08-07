import atexit
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np

import pandas as pd
import seaborn as sns

from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

import lib

args, output = lib.load_config()

program = lib.get_path(args['program'])
program_copy = program.with_name(
    program.stem + '___' + str(uuid.uuid4()).replace('-', '') + program.suffix
)
shutil.copyfile(program, program_copy)
atexit.register(lambda: program_copy.unlink())

def cv_regression():
    # initialize stratified fold split
    folds = KFold(**args['kfold'])
 
    # initialize list to document metrics across folds
    pred_list = []
    scores = []
 
    # get participant IDs
    all_ids = np.load(lib.get_path(args['data']['path']) / 'eids_all.npy')
 
    # split train data into folds, fit model within each fold
    for fold_num, (train_idxs, val_idxs) in enumerate(folds.split(X=all_ids)):
        
        # Write split data because the pipeline needs to load data from
        # disk 
        lib.write_prediction_kfold_data(lib.get_path(args['data']['path']), 
                                        train_idxs, val_idxs)

        # run NN training and predicting pipeline
        with tempfile.TemporaryDirectory() as dir_:
            dir_ = Path(dir_)
            out = dir_ / f'fold_{fold_num}'
            config_path = out.with_suffix('.toml')
            lib.dump_toml(args, config_path)
            subprocess.run(
                [
                    "python",
                    str(program_copy),
                    str(config_path),
                ],
                check=True,
            )
        
            # load predicted values on test set from disk
            preds = np.load(out / 'p_test.npy')
 
            # get R squared in the fold
            stats = lib.load_json(out / 'stats.json')
            r2 = stats['metrics'][lib.TEST]['r2']
 
        # append to list of R squared across folds
        scores.append(r2)
 
        # get participant IDs from data index
        ids = all_ids[val_idxs]
 
        # Create a dataframe for the fold with predicted values and participant IDs
        df = pd.DataFrame(index=ids)
        df['real_values'] = np.load(lib.get_path(args['data']['path']) / 'y_test.npy')
        df['predicted_values'] = preds
       
        # Append the fold dataframe to the list
        pred_list.append(df)
 
    # get average R squared across folds
    mean_r2 = np.mean(scores)
 
    # concatenate list of predictions across folds
    preds_df = pd.concat(pred_list)
 
    return mean_r2, preds_df

mean_r2, preds_df = cv_regression()
preds_df.to_csv(output / 'kfold_predictions.csv')

### metrics and plot to return for each model
# Evaluation metrics
r, pvalue = pearsonr(preds_df['real_values'], preds_df['predicted_values'])
r2 = r2_score(preds_df['real_values'], preds_df['predicted_values'])
rmse = mean_squared_error(preds_df['real_values'], preds_df['predicted_values'], squared=False)
mae = mean_absolute_error(preds_df['real_values'], preds_df['predicted_values'])
 
# plot predicted age against age
regplot = sns.regplot(
    x=preds_df['real_values'],
    y=preds_df['predicted_values'],
    scatter_kws=dict(color='midnightblue', s=10, alpha=0.8),
    line_kws=dict(color='red')
)
 
# add annotation
model_type = lib.get_path(args['program']).stem
annotation_text = f'r = {r:.4f}\nR² = {r2:.4f}\nRMSE = {rmse:.4f}\nMAE = {mae:.4f}'
plt.text(.05, .95, annotation_text, ha='left', va='top', transform=regplot.transAxes)
plt.text(.95, .1, model_type, ha='right', va='top', transform=regplot.transAxes)
 
# p-value = {pvalue:.2e}
regplot.set(xlabel='Age at recruitment (years)', ylabel='ProtAge')
fig = regplot.get_figure()
 
# save
name = output / 'regplot.png'
fig.savefig(
    name,
    dpi=600,
    transparent=False,
    bbox_inches="tight"
)
plt.close()