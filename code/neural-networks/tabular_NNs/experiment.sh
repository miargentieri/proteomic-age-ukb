model=$1
export CUDA_VISIBLE_DEVICES=$2
export PROJECT_DIR="/home/upamanyu/ProtNN/tabular_NNs"
export PYTHONPATH="$PYTHONPATH:"$PROJECT_DIR""
main_seed=3456

#########################
# Hyperparameter tuning #
#########################
cp -r data data_${model}
rm -r data_${model}/olink_proteomic
cp -r data_${model}/olink_proteomic_base data_${model}/olink_proteomic
python bin/tune.py output/olink_proteomic/${model}/tuning/${main_seed}.toml

########################################
# Train best model on all trainig data #
########################################
rm -r data_${model}/olink_proteomic
cp -r data_${model}/olink_proteomic_base data_${model}/olink_proteomic

# Set entrie training + validation as training set
cp data_${model}/olink_proteomic/N_train_val.npy data_${model}/olink_proteomic/N_train.npy
cp data_${model}/olink_proteomic/y_train_val.npy data_${model}/olink_proteomic/y_train.npy

# Set val set to the same as test set
cp data_${model}/olink_proteomic/N_test.npy data_${model}/olink_proteomic/N_val.npy
cp data_${model}/olink_proteomic/y_test.npy data_${model}/olink_proteomic/y_val.npy

# Set val_size in info.json to the same as test_size
python -c "
import json 
path = 'data_${model}/olink_proteomic/info.json'
with open(path, 'r+') as f:
    data = json.load(f)
    data['val_size'] = data['test_size']
    f.seek(0)
    json.dump(data, f, indent=4)
    f.truncate()
"

# Train and predict
# seeds=(12 34 56 78 90)
seeds=($main_seed)
for sd in "${seeds[@]}"
do
    if [ ! -d "output/olink_proteomic/${model}/tuned/" ]; then
        mkdir output/olink_proteomic/${model}/tuned/
    fi
    if [ ! -f "output/olink_proteomic/${model}/tuned/${sd}.toml" ]; then
        cp output/olink_proteomic/${model}/tuning/${main_seed}/best.toml output/olink_proteomic/${model}/tuned/${sd}.toml     
    fi

    python bin/${model}.py output/olink_proteomic/${model}/tuned/${sd}.toml 
done