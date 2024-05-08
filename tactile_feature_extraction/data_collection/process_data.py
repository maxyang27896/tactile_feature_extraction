import os
import shutil
import pandas as pd

from tactile_feature_extraction import BASE_DATA_PATH

def col_rename(path):
    df = pd.read_csv(f'{path}/targets.csv')
    names = []

    for i, _ in enumerate(df['data_name']):
        names.append(f'image_{i}.png')

    df.insert(2, 'sensor_image', names, True)
    df.to_csv(f'{path}/targets.csv', index=False)

def move_and_rename(targetPath, framePath, newPath, processed):
    df = pd.read_csv(f'{targetPath}/targets.csv')
    data_names = df['data_name'].tolist()
    file_names = os.listdir(framePath)
    i = 0
    for dataname in data_names:
        for filename in file_names:
            if processed:
                idx = filename
                idx = idx.lstrip('frame_')
                idx = idx.rstrip('_0.png')
                if int(dataname.lstrip('frame_')) == int(idx):
                    shutil.copy(f'{framePath}/{filename}', f'{newPath}/image_{i}.png')
                    i=i+1
            else:
                if int(dataname.lstrip('frame_')) == int(filename.strip('frame_.png')):
                    #shutil.copy(f'{framePath}/{filename}', f'{newPath}/image_{i}.png')
                    shutil.move(f'{framePath}/{filename}', f'{newPath}/image_{i}.png')
                    i=i+1 

# Set root path
dataPath_0 = BASE_DATA_PATH
dataPath = os.path.join(BASE_DATA_PATH, 'linshear_surface_3d', 'nanoTip')

# Define train & validation paths
trainDir = f'{dataPath}/train'
valDir = f'{dataPath}/val'

#  Define old image paths (to be moved)
framePath = f'{dataPath_0}/raw_frames'

# Define and make new image paths for:
trainImageDir = os.path.join(trainDir, 'images')
valImageDir = os.path.join(valDir, 'images')

print('renaming columns for training')
col_rename(trainDir)
col_rename(valDir)

try:
    os.mkdir(trainImageDir)
    os.mkdir(valImageDir)
except:
    pass

print('moving training images')
move_and_rename(trainDir, framePath, trainImageDir, processed=False)
print('moving validation images')
move_and_rename(valDir, framePath, valImageDir, processed=False)


