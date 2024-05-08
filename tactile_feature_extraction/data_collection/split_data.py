import os
import pandas as pd
import numpy as np

from tactile_feature_extraction import BASE_DATA_PATH

data_path = BASE_DATA_PATH
indir_name = "data"
outdir_names = ["train", "val"]
split = 0.8

targets_df = pd.read_csv(f'{data_path}/labels.csv')

# Select data
np.random.seed(0) # make predictable
inds_true = np.random.choice([True, False], size=len(targets_df), p=[split, 1-split])
inds = [inds_true, ~inds_true]

# iterate over split
for outdir_name, ind in zip(outdir_names, inds):

    indir = os.path.join(data_path, indir_name)
    outdir = os.path.join(data_path, 'linshear_surface_3d', 'nanoTip', outdir_name)

    # point image names to indir
    targets_df['image_name'] = f'{data_path}/processed_frames/' + targets_df.image_name.map(str) 

    os.makedirs(outdir, exist_ok=True)

    # populate outdir
    targets_df[ind].to_csv(os.path.join(outdir, 'targets.csv'), index=False)
    targets_df = pd.read_csv(f'{data_path}/labels.csv')