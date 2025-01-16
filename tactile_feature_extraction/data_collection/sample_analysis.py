import cv2
import os
import pickle
from scipy import signal
import numpy as np
import pandas as pd
from tqdm import tqdm

class Analyser():
    def __init__(self, path):
        self.i = None
        self.timeseriesPath = path
    
    def get_labels(self, i):
        self.i = i
        filename = f"sample_{self.i}.pkl"

        with open(os.path.join(self.timeseriesPath, filename), 'rb') as handle:
            data = pickle.load(handle)

        Fx = data["fx"]
        Fy = data["fy"]
        Fz = data["fz"]

        # Try without filtering, just take last value:
        Fx_final = Fx[-1]
        Fy_final = Fy[-1]
        Fz_final = Fz[-1]

        return Fx_final, Fy_final, Fz_final

if __name__ == '__main__':
    trial = 'test'
    parent = '/home/max-yang/Documents/Projects/tactile_feature_extraction/tactile_feature_extraction/saved_data'
    folder = f"collect_{trial}_5D_surface"
    dataPath = os.path.join(parent, folder)
    analyse = Analyser(f'{dataPath}/time_series')
    sample_range = (0,5)
    Fx = []
    Fy = [] 
    Fz = []
    missing_samples = []

    for i in tqdm(range(sample_range[0], sample_range[1])):
        try:
            # Get forces:
            forces = analyse.get_labels(i)
            Fx.append(forces[0])
            Fy.append(forces[1])
            Fz.append(forces[2])            
        except:
            missing_samples.append(i)
            Fx.append(0)
            Fy.append(0)
            Fz.append(0)
            #i = i+1
            pass
    
    for i in range(sample_range[1]):
        force_reading = [Fx[i], Fy[i], Fz[i]]

        print('Force {}: {}'.format(i, force_reading))
        
    print(f'missing samples: {missing_samples}')