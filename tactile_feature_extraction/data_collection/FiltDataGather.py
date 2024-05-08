import cv2
import pickle 
import NetFT
import os
import time

from threading import Thread, Lock

class DataGather(object):
    def __init__(self, resume, dataPath):
        # FT Sensor inits:
        self.ip = '192.168.1.1'
        self.FT18690 = NetFT.Sensor(self.ip)
        print('connected to ATI Mini! \n Calibrating...')
        self.FT18690.tare(1000)#Subtracts the mean average of 1000 samples from new data to
        print('calibration complete')

        self.dataPath = dataPath

        if resume == False:
            frame_folder = os.path.join(self.dataPath, f'raw_frames')
            os.makedirs(frame_folder, exist_ok=True)

            video_folder = os.path.join(self.dataPath, f'videos')
            os.makedirs(video_folder, exist_ok=True)

            timeseries_folder = os.path.join(self.dataPath, f'time_series')
            os.makedirs(timeseries_folder, exist_ok=True)

        self.framePath = f'{self.dataPath}/raw_frames'
        self.videoPath = f'{self.dataPath}/videos'
        self.timeseriesPath = f'{self.dataPath}/time_series'

        self.Fx = None
        self.Fy = None
        self.Fz = None

        self.Fx_list = []
        self.Fy_list = []
        self.Fz_list = []

        # TacTip inits:
        # Port
        self.cam = cv2.VideoCapture(2)
        if not self.cam.isOpened():
            raise SystemExit("Error: Could not open camera.")
        else:
            print("Camera captured successfully.")

        # Resolution
        self.cam.set(3, 640)
        self.cam.set(4, 480)
        # Exposure
        self.cam.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
        self.cam.set(cv2.CAP_PROP_EXPOSURE, 312.5)
        # Brightness
        self.cam.set(cv2.CAP_PROP_BRIGHTNESS, 64)


        self.frame = None
        self.cam_ready = False
        self.i = None

        self.sample = 0
        self.sample_list = []
        
        self.start_time = time.time()
        self.out = None

        self.t = []

        self.threadRun = False
        self.log = False
        self.cam_started = False

    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('exiting...')
        self.close()

    def start(self):
        # Start threads
        self.DataThread = Thread(None, self.readData)
        self.FTDataThread = Thread(None, self.FTStream)
        self.threadRun = True
        print(f'self.threadRun {self.threadRun}')
        
        self.FTDataThread.start()
        print('FT thread started')
        self.DataThread.start()
        print('Main thread started')
        
        time.sleep(1)
    
    def FTcalibrate(self):
        print('Calibrating ATI Mini...')
        self.FT18690.tare(1000)#Subtracts the mean average of 1000 samples from new data to calibrate
        print('calibration complete')
    
    def pause(self):
        self.log = False
        print('data threads paused')
    
    def resume(self):
        self.log = True
        print('data threads resumed')
    

    def begin_sample(self, i):
        self.i = i
        #self.out = cv2.VideoWriter(os.path.join(self.videoPath, f'sample_{self.i}.mp4'),self._fourcc, 20, (self.frame_width,self.frame_height))
        video_folder = os.path.join(self.videoPath, f'sample_{self.i}')
        os.makedirs(video_folder, exist_ok=True)
        self.videoframesPath = video_folder
        self.log = True
    
    def stop_and_write(self):
        self.log = False
        data_keys = ["t", "Fx", "Fy", "Fz","n"]
        data_lists = [self.t, self.Fx_list, self.Fy_list, self.Fz_list, self.sample_list]
        dictionary = dict(zip(data_keys,data_lists))
        
        try:
            with open(os.path.join(self.timeseriesPath, f'sample_{self.i}.pkl'), 'wb') as handle:
                pickle.dump(dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
            #FOR SINGLE FRAME CAP:
            time.sleep(0.1)
            filenames = os.listdir(self.videoframesPath)
            max_value = max(int(filename.strip('frame_.png')) for filename in filenames)
            i = 0
            while i < max_value:
                os.remove(f'{self.videoframesPath}/frame_{i}.png')
                i=i+1
            i=0
        except:
            pass
        
        # Reset variables:
        self.t = []
        self.Fx_list = []
        self.Fy_list = []
        self.Fz_list = []
        self.sample_list = []
        self.sample = 0

        
    def FTStream(self):
        # Captures FT data while self.log = True
        while self.threadRun:
            if self.cam_ready and self.log:
                try:
                    dataCounts = self.FT18690.getMeasurement()#Get a single sample of all data Fx, Fy, Fz, Tx, Ty, Tz returned as list[6]
                    self.Fx = (dataCounts[0]/1000000)
                    self.Fy = -1*(dataCounts[1]/1000000)
                    self.Fz = -1*(dataCounts[2]/1000000)

                    if self.log:
                        self.Fx_list.append(self.Fx)
                        self.Fy_list.append(self.Fy)
                        self.Fz_list.append(self.Fz)
                        self.sample_list.append(self.sample)
                except:
                    pass
                

    def readData(self):
        # Captures image data while self.log = True
        while self.threadRun:
            if self.log:
                try:
                    self.cam_ready = True
                    self.t.append(time.time())
                    success, self.frame = self.cam.read()
                    self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
                    #self.frame = cv2.resize(self.frame, (320, 180))
                    cv2.imwrite(os.path.join(self.videoframesPath, f'frame_{self.sample}.png'), self.frame)
                    #cv2.imshow("capture", self.frame) # Display video stream
                    cv2.waitKey(1)

                    if success == False:
                        break

                    self.sample = self.sample +1
                except:
                    self.sample = self.sample + 1
                    pass

        
    def returnData(self):
        return [self.t, self.Fx_list, self.Fy_list, self.Fz_list]
            
    def stop(self):
        if self.threadRun:
            self.threadRun = False

            self.FTDataThread.join()
            self.DataThread.join()
            print("Threads joined successfully")

            self.cam.release()
            print("Camera resources released")


    def close(self):
        self.stop()

lock = Lock()
