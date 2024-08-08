# ProtNN

This is a repository for testing tabular NN architectures for proteomic data.

## Installation and Environment Setup

To set up the environment to run the code, please refer to:

- [Overview](https://github.com/yandex-research/tabular-dl-revisiting-models/blob/main/README.md#2-overview)
- [Environment Setup](https://github.com/yandex-research/tabular-dl-revisiting-models/blob/main/README.md#3-setup-the-environment)

## Directory structure

The **tabular_NNs folder** contains code to run the following models:
- MLP
- ResNet
- Transformer 

The **TabR folder** contains the code to run TabR.

`Note: The Transformer model was not executed because of very high computational
time required to run cross validation`

## Preparing Data

Within the ```TabR``` and ```tabular_NNs``` folders, the data should be present
in the following structure: 

```
- tabular_NNs
    - data
        - olink_proteomic
            - N_all.npy
            - N_train_val.npy
            - N_test.npy
            - y_all.npy
            - y_train_val.npy
            - y_test.npy
            - feature_names.txt
            - info.json

- TabR
    - data
        - olink_proteomic
            - X_num_all.npy
            - X_num_train_val.npy
            - X_num_test.npy
            - Y_all.npy
            - Y_train_val.npy
            - Y_test.npy
            - feature_names.txt
            - info.json
```

The N_* and X_num_* files are numpy arrays containing the proteomic data to be
input to the neural networks. The y_* and Y_* files contain the age to be
predicted. The feature names corresponding to the colums in the data should be
stored in feature_names.txt.

The struture of the ```info.json``` file is as follows:

```
{
    "name": "oLink_proteomic",
    "id": "olink_proteomic",
    "task_type": "regression",
    "n_num_features": <num_proteins>,
    "n_cat_features": 0,
    "train_size": <num_train_samples>,
    "val_size": 0,
    "test_size": <num_test_samples>,
    "n_bin_features": 0
}
```

The validation size is initially left as 0 because it will be determined when
running the 5-fold CV.

Also, create a copy of the data folder:

```Bash
cp -r tabular_NNs/data/olink_proteomic tabular_NNs/data/olink_proteomic_base
cp -r TabR/data/olink_proteomic TabR/data/olink_proteomic_base
```


## Tuning and Training

To run the analysis, use the **experiment.sh** file in tabular_NNs and TabR
folders. 

### MLP and Resnet

```Bash
bash tabular_NNs/experiment.sh model_name GPU_num
```

where ```model_name``` and ```GPU_num``` are the type of model you want to run
and the gpu device number you want to use. If you do not want to use a GPU,
leave this blank. However, it will take a long time to run without a GPU. An
example execution to run the MLP model using gpu 0 would be:

```Bash
bash tabular_NNs/experiment.sh mlp 0
```

The two steps exeucted by this code are:

1. Run the hyperparameter tuning. 

2. Train the final models on the entire training set and
   predict on the testing set. 

The tuned model will be stored in
```outputs/olink_proteomic/<model_name>/tuned/3456/```. The trained model is
stored in ```checkpoint.pt```.

### TabR

```Bash
bash TabR/experiment.sh tabr GPU_num
```

where ```GPU_num``` is the gpu device number you want to use. If you do not want
to use a GPU, leave this blank. However, it will take a long time to run without
a GPU. An example execution to run the TabR model using gpu 0 would be:

```Bash
bash tabular_NNs/experiment.sh tabr 0
```

The two steps exeucted by this code are:

1. Run the hyperparameter tuning. 

2. Train the final models on the entire training set and
   predict on the testing set.

The tuned model will be stored in
```exp/tabr/olink_proteomic/3456-evaluation/```. The trained model is
stored in ```3456/checkpoint.pt``` and the model config will be stored in ```3456.toml```.

## Final predictions and plots

1. Copy data into the folder:

    ```Bash
    cp -r data_mlp/olink_proteomic final_models/olink_proteomic
    ```

2. Create a checkpoints and output folder in the final_models folder to store
   model checkpoints and results:

    ```Bash
    mkdir final_models/ckpts
    mkdir final_models/output
    ```

3. Copy the checkpoints and configs into the folder: 

    ```Bash
    # MLP
    cp tabular_NNs/outputs/olink_proteomic/mlp/tuned/3456/checkpoint.pt final_models/ckpts/MLP_checkpoint.pt

    # ResNet
    cp tabular_NNs/outputs/olink_proteomic/resnet/tuned/3456/checkpoint.pt final_models/ckpts/ResNet_checkpoint.pt

    # TabR
    cp exp/tabr/olink_proteomic/3456-evaluation/3456/checkpoint.pt final_models/ckpts/TabR_checkpoint.pt
    cp exp/tabr/olink_proteomic/3456-evaluation/3456.toml final_models/ckpts/TabR_config.toml
    ```

4. Run the code to generate plots and shap values:

    ```Bash
    python final_models/tuned_model_predictions.py
    ```

## Contributions

- Code written and run by [Upamanyu Ghose](https://titoghose.github.io/)
- The code in this repository is largely adapted from:
  - [yandex-research/tabular-dl-revisiting-models](https://github.com/yandex-research/tabular-dl-revisiting-models)
  - [yandex-research/tabular-dl-tabr](https://github.com/yandex-research/tabular-dl-tabr)
