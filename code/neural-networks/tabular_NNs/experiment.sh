model=$1
export CUDA_VISIBLE_DEVICES=$2
export PROJECT_DIR="/home/upamanyu/ProtNN/tabular_NNs"
export PYTHONPATH="$PYTHONPATH:"$PROJECT_DIR""
main_seed=3456

#########################
# Hyperparameter tuning #
#########################
# cp -r data data_${model}
# rm -r data_${model}/olink_proteomic
# cp -r data_${model}/olink_proteomic_base data_${model}/olink_proteomic
# python bin/tune.py output/olink_proteomic/${model}/tuning/${main_seed}.toml

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

#################################################
# Predictions on entire dataset using 5-fold CV #
#################################################
# rm -r data_${model}/olink_proteomic
# cp -r data_${model}/olink_proteomic_base data_${model}/olink_proteomic

# best_tuning_toml=output/olink_proteomic/${model}/tuning/${main_seed}/best.toml
# final_folder=output/olink_proteomic/${model}/final
# final_toml=${final_folder}/${main_seed}.toml

# mkdir $final_folder 
# cp $best_tuning_toml $final_toml

# # Write params to final toml
# temp_file=$(mktemp)
# echo "program = 'bin/"$model".py'" > $temp_file
# echo "" >> $temp_file
# cat "$final_toml" >> "$temp_file"
# mv "$temp_file" "$final_toml"

# echo "" >> $final_toml
# echo "[kfold]" >> $final_toml
# echo "n_splits = 5" >> $final_toml
# echo "shuffle = true" >> $final_toml
# echo "random_state = "$main_seed"" >> $final_toml

# python bin/kfold_cv_predictions.py $final_toml