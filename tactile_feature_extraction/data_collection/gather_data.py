import cv2
import pickle 
import Pyro4
import os
import time
import socket, struct
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from time import sleep
#from tactile_feature_extraction.utils.utils_ft_sensor import Sensor

from threading import Thread, Lock

class DataGatherer(object):
	def __init__(self, resume, dataPath, time_series, display_image, FT_ip, resize):
		self.resize = resize
		# FT Sensor inits:
		self.ip = FT_ip
		self.port = 49152
		self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
		self.sock.connect((FT_ip, self.port))
		self.mean = [0] * 6
		self.stream = False
		self.newtons_data = None

		# Path inits:
		self.dataPath = dataPath
		self.time_series = time_series 
		self.display_image = display_image

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
		self.cam = cv2.VideoCapture(0)
		if not self.cam.isOpened():
			raise SystemExit("Error: Could not open camera.")
		else:
			print("Camera opened successfully.")

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

		# Sampling inits:
		self.sample = 0 # Sample number
		self.sample_list = []
		
		self.start_time = time.time()
		self.out = None

		self.t = []

		self.threadRun = False
		self.log = False

	def __enter__(self):
		return self
	
	def __exit__(self, exc_type, exc_val, exc_tb):
		print('exiting...')
		self.close()

	def start(self):
		self.tare(n=1000) # Calibrate sensor

		# Start main data logging threads:
		self.imageThread = Thread(None, self.image_worker, daemon=True) 
		self.startStreaming() # Starts FT stream
		self.threadRun = True
		print(f'self.threadRun {self.threadRun}')
		self.imageThread.start() # Starts camera stream
		
		time.sleep(1)

	def begin_sample(self, i):
		''' Begins a new sample by making a directory for video data
			according to the sample number.
		'''
		self.i = i
		video_folder = os.path.join(self.videoPath, f'sample_{self.i}')
		os.makedirs(video_folder, exist_ok=True)
		self.videoframesPath = video_folder
		self.log = True
	
	def avg_force(self, data, t):
		''' Calculates the average force per frame for a given sample 
		    and returns as a dictionary.
		'''
		# Transpose the input data to match the expected format for DataFrame creation
		n, fx, fy, fz = data
		structured_data = list(zip(n, fx, fy, fz))
		
		# Create a DataFrame from the structured data
		df = pd.DataFrame(structured_data, columns=['frame', 'fx', 'fy', 'fz'])
		
		# Compute the mean values of fx, fy, fz for each value of n
		mean_df = df.groupby('frame').mean().reset_index()
	
		# Adjust the length of t to match the length of mean_df
		if len(t) < len(mean_df):
			t.extend([np.nan] * (len(mean_df) - len(t)))
		elif len(t) > len(mean_df):
			t = t[:len(mean_df)]
    
		# Add the adjusted 't' column to the DataFrame
		mean_df['t'] = t
		
		# Convert the final DataFrame to a dictionary
		result_dict = mean_df.to_dict(orient='list')
		
		return result_dict
	
	def stop_and_write(self):
		''' Stop logging data at the end of a sample and save the information.
		'''
		self.log = False
		data_lists = [self.sample_list, self.Fx_list, self.Fy_list, self.Fz_list]
		processed_data = self.avg_force(data_lists, self.t)
		
		try:
			# Save time-series force data, after finding the average force for each frame
			with open(os.path.join(self.timeseriesPath, f'sample_{self.i}.pkl'), 'wb') as handle:
				pickle.dump(processed_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
			
			# For single frame capture (i.e. if not intending to use data for time-series analysis):
			if self.time_series == False:
				time.sleep(0.1)
				filenames = os.listdir(self.videoframesPath)
				max_value = max(int(filename.strip('frame_.png')) for filename in filenames)
				i = 0
				while i < max_value:
					# Removes video frames apart from the last frame:
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

############################# FT SENSOR FUNCTIONS ##################################
	def send(self, command, count = 0):
		'''Send a given command to the Net F/T box with specified sample count.

		This function sends the given RDT command to the Net F/T box, along with
		the specified sample count, if needed.

		Args:
			command (int): The RDT command.
			count (int, optional): The sample count to send. Defaults to 0.
		'''
		header = 0x1234
		message = struct.pack('!HHI', header, command, count)
		self.sock.send(message)

	def receive(self):
		'''Receives and unpacks a response from the Net F/T box.

		This function receives and unpacks an RDT response from the Net F/T
		box and saves it to the data class attribute.

		Returns:
			list of float: The force and torque values received. The first three
				values are the forces recorded, and the last three are the measured
				torques.
		'''
		rawdata = self.sock.recv(1024)
		data = struct.unpack('!IIIiiiiii', rawdata)[3:]
		self.data = [data[i] - self.mean[i] for i in range(6)]
		if self.log and self.cam_ready:
			self.to_newtons(self.data)
		return self.data
	
	def to_newtons(self, data):
		Fx = data[0]/1000000
		Fy = -1*(data[1]/1000000)
		Fz = -1*(data[2]/1000000)
		
		lock.acquire()
		self.Fx_list.append(Fx)
		self.Fy_list.append(Fy)
		self.Fz_list.append(Fz)
		self.sample_list.append(self.sample)
		lock.release()

	def tare(self, n = 10):
		'''Tare the sensor.

		This function takes a given number of readings from the sensor
		and averages them. This mean is then stored and subtracted from
		all future measurements.

		Args:
			n (int, optional): The number of samples to use in the mean.
				Defaults to 10.

		Returns:
			list of float: The mean values calculated.
		'''
		self.mean = [0] * 6
		self.getMeasurements(n = n)
		mean = [0] * 6
		for i in range(n):
			self.receive()
			for i in range(6):
				mean[i] += self.measurement()[i] / float(n)
		self.mean = mean
		return mean

	def zero(self):
		'''Remove the mean found with `tare` to start receiving raw sensor values.'''
		self.mean = [0] * 6

	def receiveHandler(self):
		'''A handler to receive and store data.'''
		while self.stream:
			self.receive()

	def getMeasurement(self):
		'''Get a single measurement from the sensor

		Request a single measurement from the sensor and return it. If
		The sensor is currently streaming, started by running `startStreaming`,
		then this function will simply return the most recently returned value.

		Returns:
			list of float: The force and torque values received. The first three
				values are the forces recorded, and the last three are the measured
				torques.
		'''
		self.getMeasurements(1)
		self.receive()
		return self.data

	def measurement(self):
		'''Get the most recent force/torque measurement

		Returns:
			list of float: The force and torque values received. The first three
				values are the forces recorded, and the last three are the measured
				torques.
		'''
		return self.data

	def getForce(self):
		'''Get a single force measurement from the sensor

		Request a single measurement from the sensor and return it.

		Returns:
			list of float: The force values received.
		'''
		return self.getMeasurement()[:3]

	def force(self):
		'''Get the most recent force measurement

		Returns:
			list of float: The force values received.
		'''
		return self.measurement()[:3]

	def getTorque(self):
		'''Get a single torque measurement from the sensor

		Request a single measurement from the sensor and return it.

		Returns:
			list of float: The torque values received.
		'''
		return self.getMeasurement()[3:]

	def torque(self):
		'''Get the most recent torque measurement

		Returns:
			list of float: The torque values received.
		'''
		return self.measurement()[3:]

	def startStreaming(self, handler = True):
		'''Start streaming data continuously

		This function commands the Net F/T box to start sending data continuously.
		By default this also starts a new thread with a handler to save all data
		points coming in. These data points can still be accessed with `measurement`,
		`force`, and `torque`. This handler can also be disabled and measurements
		can be received manually using the `receive` function.

		Args:
			handler (bool, optional): If True start the handler which saves data to be
				used with `measurement`, `force`, and `torque`. If False the
				measurements must be received manually. Defaults to True.
		'''
		self.getMeasurements(0)
		if handler:
			self.stream = True
			self.thread = Thread(target = self.receiveHandler)
			self.thread.daemon = True
			self.thread.start()

	def getMeasurements(self, n):
		'''Request a given number of samples from the sensor

		This function requests a given number of samples from the sensor. These
		measurements must be received manually using the `receive` function.

		Args:
			n (int): The number of samples to request.
		'''
		self.send(2, count = n)

	def stopStreaming(self):
		'''Stop streaming data continuously

		This function stops the sensor from streaming continuously as started using
		`startStreaming`.
		'''
		self.stream = False
		sleep(0.1)
		self.send(0)

	def image_worker(self):
		# Worker thread which captures image data while self.log = True
		while self.threadRun:
			if self.log:
				try:
					self.cam_ready = True
					self.t.append(time.time())
					success, self.frame = self.cam.read()
					self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
					
					if self.display_image:
						cv2.imshow("capture", self.frame) # Display video stream
					
					if self.resize[0]:
						self.frame = cv2.resize(self.frame, (self.resize[1]))
					cv2.imwrite(os.path.join(self.videoframesPath, f'frame_{self.sample}.png'), self.frame)
					
					cv2.waitKey(1)

					if success == False:
						print('No image data')
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

			self.imageThread.join()
			self.stopStreaming()
			print("Main threads joined successfully")

	def close(self):
		self.stop()

lock = Lock()
