import glob
import os
import csv
import numpy as np
import pdb
from tqdm import tqdm

csv_filename = os.path.join('data/Hypersim', "depth_stats.csv")
assert(os.path.exists(csv_filename))

# read the csv file first
with open(csv_filename, encoding="UTF-8") as file:
    reader = csv.DictReader(file)
    metadata = {}
    for row in reader:
        for column, value in row.items():
            metadata.setdefault(column, []).append(value)

# not only train
split = np.array(metadata["split"])
train_index = split=="test"
nan_ratio= [float(x) for x in metadata["nan_ratio"]]
nan_ratio = np.array(nan_ratio)[train_index]
print(len(nan_ratio))
print(np.sum(nan_ratio<0.01))
print(np.sum(nan_ratio<0.02))
print(np.sum(nan_ratio<0.03))
print(np.sum(nan_ratio<0.04))
print(np.sum(nan_ratio<0.05))
print(np.sum(nan_ratio<0.1))
print(np.sum(nan_ratio<0.2))
print(np.sum(nan_ratio<0.3))

###
# threshold: 0.04
# 59543
# 51403
# 53185
# 54161
# 54983
# 55614
# 57372
# 58793
# 59239
# ###
