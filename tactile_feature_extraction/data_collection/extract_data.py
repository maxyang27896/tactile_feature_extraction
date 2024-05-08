import os
import shutil
from PIL import Image
import pandas as pd
from tqdm import tqdm

from sample_analysis import Analyser

from tactile_feature_extraction import BASE_DATA_PATH

def try_open_image(image_path):
    with Image.open(image_path) as img:
        img.verify()  # Verify the integrity of the file.
        return True

dataPath = BASE_DATA_PATH
videopath = os.path.join(dataPath, 'videos')
framePath = f'{dataPath}/raw_frames'
analyse = Analyser(f'{dataPath}/time_series')

print('extracting data...')
missing_samples = []
missing_frames = []
Fx = []
Fy = [] 
Fz = []
frames = []
sample_range = (0, 3000)
n_samples = sample_range[1]
for i in tqdm(range(sample_range[0], sample_range[1])):
    try:
        # Get frame:
        filename = os.listdir(f'{videopath}/sample_{i}')
        
        # Try opening image
        img_path = f'{videopath}/sample_{i}/{filename[0]}'
        try_open_image(img_path)

        # Get forces:
        forces = analyse.get_data_and_labels(i)
        Fx.append(forces[0])
        Fy.append(forces[1])
        Fz.append(forces[2])
        frames.append(f'frame_{i}')
        
        # Check frame and move to new location
        shutil.copy(img_path, f'{framePath}/frame_{i}.png')
    except:
        missing_samples.append(i)
        Fx.append(0)
        Fy.append(0)
        Fz.append(0)
        frames.append(f'frame_{i}')
        #i = i+1
        pass

# Create the data frame 
force_df = pd.DataFrame(list(zip(frames,Fx,Fy,Fz)), columns = ['data_name_force','Fx','Fy','Fz'])
torque_data = pd.DataFrame(list(zip([0]*n_samples,
                                    [0]*n_samples,
                                    [0]*n_samples)), columns = ['Tx','Ty','Tz'])

df = pd.read_csv(f'{dataPath}/targets.csv')
final_df = pd.concat([df, force_df, torque_data], axis=1)

# Filter out data 
for sample_name in missing_samples:
    ind = final_df[(final_df['pose_id'] == (sample_name+1))].index
    final_df.drop(ind, inplace=True)

final_df.to_csv(f"{dataPath}/labels.csv")
print('done')
print(f'{len(missing_samples)} missing samples: {missing_samples}')
print(f'{len(missing_frames)} missing frames: {missing_frames}')