from typing import Dict, Optional, Tuple, Union

import delu
import numpy as np
import pandas as pd
import seaborn as sns

# import shap
import tomli
import torch
import tqdm
import zero
from matplotlib import pyplot as plt
from models import MLP
from models import Model as TabR
from models import ResNet, Transformer
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# TabR sepecific functions
_TOML_CONFIG_NONE = '__null__'
def _process_toml_config(data, load) -> dict:
    if load:
        # replace _TOML_CONFIG_NONE with None
        condition = lambda x: x == _TOML_CONFIG_NONE  # noqa
        value = None
    else:
        # replace None with _TOML_CONFIG_NONE
        condition = lambda x: x is None  # noqa
        value = _TOML_CONFIG_NONE

    def do(x):
        if isinstance(x, dict):
            return {k: do(v) for k, v in x.items()}
        elif isinstance(x, list):
            return [do(y) for y in x]
        else:
            return value if condition(x) else x

    return do(data)  # type: ignore[code]

def get_Xy(X: torch.Tensor, y: torch.Tensor, idx) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    batch = ({'num': X}, y)
    return (
        batch
        if idx is None
        else (
            {k: v[idx] for k, v in batch[0].items()}, 
            batch[1][idx] if batch[1] is not None else None
        )
    )

def load_model_from_config(model_name:str, model_path:str, device:torch.device) -> torch.nn.Module:
    with open(f'ckpts/{model_name}_config.toml', 'rb') as f:
        C = _process_toml_config(tomli.load(f), True)

    model = eval(model_name)(**C['model'])
    state_dict = torch.load(model_path, map_location=device)['model']

    # If models with run with nn.DataParallel, the state_dict will have
    # 'module.' prepended to all the keys. 
    new_state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}
    model.load_state_dict(new_state_dict)

    return model

# Prediction function for all NNs
def nn_predict(model_name:str, model_path:str, data: dict,
               device:torch.device='cpu', batch_size:int=4028,
               return_tensor:bool=False) -> Union[torch.Tensor, np.ndarray]:
    
    try:
        model = torch.load(model_path, map_location=device)['model_obj']
    except KeyError:
        model = load_model_from_config(model_name, model_path, device)
    model = model.to(device)

    if isinstance(data['X_test'], np.ndarray):
        data = {k: torch.from_numpy(v).float() if v is not None else v 
                for k, v in data.items()}
        
    predictions = []
    model.eval()
    for i in tqdm.tqdm(range(0, len(data['X_test']), batch_size), 
                       desc='Batch', leave=False):
        # TabR requires the training data to be passed along with the
        # test data because it uses the training data while making
        # predictions on the testing data
        if isinstance(model, TabR):
            batch_x, _ = get_Xy(
                data['X_test'].to(device), 
                None, 
                range(i, min(i+batch_size, len(data['X_test']))))
            
            candidate_x, candidate_y = get_Xy(
                data['X_train'].to(device), 
                data['y_train'].to(device), 
                None)
            
            preds = model(
                x_ = batch_x, 
                y = None, 
                candidate_x_ = candidate_x, 
                candidate_y = candidate_y,
                context_size = 96,
                is_train = False).detach().cpu()
        else:
            batch = data['X_test'][i:i+batch_size].to(device)
            preds = model(batch).detach().cpu()
        
        predictions.append(preds)
    
    predictions = torch.concat(predictions)
    
    # transformer predcitions need to be scaled
    if isinstance(model, Transformer):
        with open(f'ckpts/{model_name}_config.toml', 'rb') as f:
            C = _process_toml_config(tomli.load(f), True)
        predictions = predictions * C['data']['y_std'] + C['data']['y_mean']
    
    if return_tensor:
        return predictions    
    else:
        return predictions.numpy()

def shap_plots(model_path:str, data: dict, feature_names:list, 
               device:torch.device='cpu') -> plt.Axes:
    
    try:
        model = torch.load(model_path, map_location=device)['model_obj']
    except KeyError:
        model = load_tabr_model(model_path)
    model = model.to(device)

    if isinstance(data['X_test'], np.ndarray):
        data = {k: torch.from_numpy(v).float().to(device) if v is not None else v 
                for k, v in data.items()}

    # Add a layer to the model that reshapes the output to [batch_size,1]
    model = torch.nn.Sequential(
                model,
                torch.nn.Unflatten(0, (-1, 1)))
        
    # Code to generate SHAP plots, given the model and data
    random_sample = torch.randint(0, len(data['X_train']), (1000,))
    explainer = shap.DeepExplainer(model, data['X_train'][random_sample])
    
    # data['X_test'] = data['X_test'][:10]
    shap_values = explainer.shap_values(data['X_test'])
    shap.summary_plot(shap_values, 
                      pd.DataFrame(data['X_test'].cpu().numpy(), 
                                   columns=feature_names),
                      show=False, 
                      plot_size=(15, 12))
    
    return plt.gca()


if __name__=='__main__':
    X = np.load('olink_proteomic/N_test.npy')
    y = np.load('olink_proteomic/y_test.npy')
    X_train = np.load('olink_proteomic/N_train.npy')
    y_train = np.load('olink_proteomic/y_train.npy')
    data = {
        'X_test': X, 
        'y_test': None, 
        'X_train': X_train, 
        'y_train': y_train}
    
    #############################################
    # metrics and plot to return for each model #
    #############################################
    random_seed = 3456
    tuned_model_paths = {
        'MLP': 'ckpts/MLP_checkpoint.pt',
        'ResNet': 'ckpts/ResNet_checkpoint.pt',
        'TabR': 'ckpts/TabR_checkpoint.pt',
        'Transformer': 'ckptsTransformer_checkpoint.pt'
    }

    for model_name, model_path in tuned_model_paths.items():
        if model_name == 'TabR':
            delu.random.seed(random_seed)
        else:
            zero.set_randomness(random_seed)
            
        predictions = np.squeeze(nn_predict(model_name, model_path, data, 
                                            device=torch.device('cpu'), 
                                            batch_size=8 if model_name=='Transformer' else 4028))
        preds_df = pd.DataFrame({'real_values': y, 'predicted_values': predictions})
        
        # Evaluation metrics
        r, pvalue = pearsonr(preds_df['real_values'], preds_df['predicted_values'])
        r2 = r2_score(preds_df['real_values'], preds_df['predicted_values'])
        rmse = mean_squared_error(preds_df['real_values'], preds_df['predicted_values'], squared=False)
        mae = mean_absolute_error(preds_df['real_values'], preds_df['predicted_values'])
        print(f'{model_name} MAE: {mae:.4f} RMSE: {rmse:.4f} R2: {r2:.4f}')

        # plot predicted age against age
        regplot = sns.regplot(
            x=preds_df['real_values'],
            y=preds_df['predicted_values'],
            scatter_kws=dict(color='midnightblue', s=10, alpha=0.8),
            line_kws=dict(color='red')
        )
        
        # add annotation
        model_type = model_name
        annotation_text = f'r = {r:.4f}\nR² = {r2:.4f}\nRMSE = {rmse:.4f}\nMAE = {mae:.4f}'
        plt.text(.05, .95, annotation_text, ha='left', va='top', transform=regplot.transAxes)
        plt.text(.95, .1, model_type, ha='right', va='top', transform=regplot.transAxes)
        
        # p-value = {pvalue:.2e}
        regplot.set(xlabel='Age at recruitment (years)', ylabel='ProtAge')
        fig = regplot.get_figure()
        
        # save
        name = f'output/regplot_UKB_{model_name}.png'
        fig.savefig(
            name,
            dpi=600,
            transparent=False,
            bbox_inches="tight"
        )
        plt.close()


    ###############
    # SHAP  plots #
    ###############
    tuned_model_paths = {
        'MLP': 'ckpts/MLP_checkpoint.pt',
        'ResNet': 'ckpts/ResNet_checkpoint.pt',
        'TabR': 'ckpts/TabR_checkpoint.pt',
        'Transformer': 'ckptsTransformer_checkpoint.pt'
    }
    with open('olink_proteomic/feature_names.txt', 'r') as f:
        olink_names = f.read().strip('\n').splitlines()

    preds_df = pd.DataFrame({'real_values': y})
    
    for model_name, model_path in tuned_model_paths.items():
        fig = plt.figure()
        ax_cnt = 0
        
        random_seed = int(model_path.split('_')[-1].split('.')[0])
        zero.set_randomness(random_seed)
        
        predictions = np.squeeze(nn_predict(model_path, data, device=torch.device('cpu')))
        preds_df[f'{model_name}'] = predictions

        ax = fig.add_subplot(2, 3, ax_cnt+1)
        shap_plots(model_path, data, feature_names=olink_names, 
                   device=torch.device(5))
        ax.set_title(f'Random seed: {random_seed}')
    
        # SHAP plot
        name = f'output/SHAPplot_UKB_{model_name}.png'
        fig.savefig(
            name,
            dpi=600,
            transparent=False,
            bbox_inches="tight"
        )
        plt.close()