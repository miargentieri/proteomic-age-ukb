model=$1
export CUDA_VISIBLE_DEVICES=$2
seed=3456

#########################
# Hyperparameter tuning #
#########################
# rm -r data/olink_proteomic
# cp -r data/olink_proteomic_base data/olink_proteomic
# python bin/go.py --n_seeds 1 exp/${model}/olink_proteomic/${seed}-tuning.toml

########################################
# Train best model on all trainig data #
########################################
rm -r data/olink_proteomic
cp -r data/olink_proteomic_base data/olink_proteomic

# Set entrie training + validation as training set
cp data/olink_proteomic/X_num_train_val.npy data/olink_proteomic/X_num_train.npy
cp data/olink_proteomic/Y_train_val.npy data/olink_proteomic/Y_train.npy

# Set val set to the same as test set
cp data/olink_proteomic/X_num_test.npy data/olink_proteomic/X_num_val.npy
cp data/olink_proteomic/Y_test.npy data/olink_proteomic/Y_val.npy

# Set val_size in info.json to the same as test_size
python -c "
import json 
path = 'data/olink_proteomic/info.json'
with open(path, 'r+') as f:
    data = json.load(f)
    data['val_size'] = data['test_size']
    f.seek(0)
    json.dump(data, f, indent=4)
    f.truncate()
"

python bin/evaluate.py \
    --n_seeds 1 \
    --function bin.tabr.main \
    exp/${model}/olink_proteomic/${seed}-evaluation 