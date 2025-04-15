import torch
import pandas
import os

os.makedirs(os.path.join("..","data"),exist_ok=True)

data_file=os.path.join("..","data","data.csv")

with open(data_file,'w') as f:
