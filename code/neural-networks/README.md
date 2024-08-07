# ProtNN

This is a repository for testing tabular NN architectures for proteomic data.
The code in this repository is largely adapted from [yandex-research/tabular-dl-revisiting-models](https://github.com/yandex-research/tabular-dl-revisiting-models)
with some task specific modifications. 

## Installation and Environment Setup

To set up the environment to run the code, please refer to:

- [Overview](https://github.com/yandex-research/tabular-dl-revisiting-models/blob/main/README.md#2-overview)
- [Environment Setup](https://github.com/yandex-research/tabular-dl-revisiting-models/blob/main/README.md#3-setup-the-environment)

## Usage

To run the project, copy the following lines into a bash script:

```bash
export PROJECT_DIR="/abs/path/to/project_dir"

# Comment out line if GPU is not available
export CUDA_VISIBLE_DEVICES="0" 
export PYTHONPATH="$PYTHONPATH:"$PROJECT_DIR"" 

python bin/tune.py output/random_proteomic/ft_transformer/tuning/0.toml
```

## Contributing

Details about how to contribute to the project.

## License

Information about the project's license.