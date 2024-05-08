import cv2
import os
import pickle
from scipy import signal
import numpy as np
import pandas as pd
from tqdm import tqdm

class Analyser():
    def __init__(self, path):
        #self.run = run
        self.i = None
        self.timeseriesPath = path

    def filter(self, fc, fps, zipped_data):
        # Takes data of the form [(sample_number, force), (sample_number, force) ... (sample_number, force)] and calculates the mean for each sample number:
        df = pd.DataFrame(zipped_data) # Create a dataframe from zipped data
        df = df.groupby(0).mean() # Group by common sample number and find the mean of each
        means = df[1].tolist() # Return means as a list, will be ordered by sample number
        w = fc / (fps / 2) # Normalize the frequency
        b, a = signal.butter(5, w, 'low')
        filtered = signal.filtfilt(b, a, means)
        
        return filtered[-1]

    def load_video(self, video_path):
        # Create a video capture object
        cap = cv2.VideoCapture(video_path)

        # Check if the video was opened successfully
        if not cap.isOpened():
            raise IOError("Could not open the video file.")

        # Read the video frame-by-frame
        frames = []
        while True:
            # Read the next frame
            ret, frame = cap.read()

            # If we reached the end of the video, break the loop
            if not ret:
                break

            # Add the frame to the list of frames
            frames.append(np.squeeze(frame))

        # Release the video capture object
        cap.release()

        return frames
    
    def get_data_and_labels(self, i):
        self.i = i
        filename = f"sample_{self.i}.pkl"
        #vid_path = f"C:/Users/c28-ford/Project/FT/data/collect_{self.run}_5D_surface/videos/sample_{self.i}.mp4"
        #frame_path = f'C:/Users/c28-ford/Project/FT/data/collect_{self.run}_5D_surface/frames'
        #frames = self.load_video(vid_path)

        with open(os.path.join(self.timeseriesPath, filename), 'rb') as handle:
            data = pickle.load(handle)

        t = data["t"]
        n = data["n"]
        Fx = data["Fx"]
        Fy = data["Fy"]
        Fz = data["Fz"]
        t0 = t[0]
        fc = 2 #filter cutoff freq

        for timestamp in range(len(t)):
            t[timestamp] = t[timestamp] - t0

        fps = len(t)/t[-1]

        Fx_zip = list(zip(n,Fx))
        Fy_zip = list(zip(n,Fy))
        Fz_zip = list(zip(n,Fz))

        Fx_final = self.filter(fc, fps, Fx_zip)
        Fy_final = self.filter(fc, fps, Fy_zip)
        Fz_final = self.filter(fc, fps, Fz_zip)
        #final_frame = frames[-1]

        #cv2.imwrite(os.path.join(frame_path, f'frame_{self.i}.png'), final_frame)

        return Fx_final, Fy_final, Fz_final

        
if __name__ == '__main__':
    trial = 'test'
    parent = '/home/max-yang/Documents/Projects/tactile_forces/collected_data/'
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
            forces = analyse.get_data_and_labels(i)
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
        force_reading = [Fx[i][-1], Fy[i][-1], Fz[i][-1]]

        print('Force {}: {}'.format(i, force_reading))
        
    print(f'missing samples: {missing_samples}')