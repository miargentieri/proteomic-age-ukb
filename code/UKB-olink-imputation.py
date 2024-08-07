import miceforest as mf
import pandas as pd
import datetime as dt

random_seed = 3456

# Load protein data
data_path = '.../olink_data_wide_oct_30_2023.csv'
data = pd.read_csv(data_path)
data = data[data.columns[data.columns != 'X']]

# Create a dictionary of variables use to impute
exclude = ['eid', 'olink_batch', 'olink_plate', 'GLIPR1', 'NPM1', 'PCOLCE']
dont_impute = ['eid', 'olink_batch', 'olink_plate']
column_dict = {col: [other_col for other_col in data.columns if other_col != col and other_col not in exclude] for col in data.columns if col not in dont_impute}

# run miceforest imputation on multiple cores
kds = mf.ImputationKernel(
  data,
  datasets=1,
  variable_schema=column_dict,
  random_state=random_seed
)

# run
kds.mice(
  iterations=5,
  n_jobs=-1, 
  verbose=True
)

# get the completed dataframe from the miceforest object
olink_data_imputed = kds.complete_data()

name = ".../olink_data_imputed_nov_18_2023.csv"
olink_data_imputed.to_csv(name, index=False)
