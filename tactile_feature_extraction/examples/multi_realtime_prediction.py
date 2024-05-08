import os
import time
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from threading import Thread

import torch
from torch.autograd import Variable

from tactile_feature_extraction import BASE_MODEL_PATH
from tactile_feature_extraction.utils.utils_image_processing import SimpleSensor
from tactile_feature_extraction.utils.utils_image_processing import list_camera_sources

from tactile_feature_extraction.utils.image_transforms import process_image

from tactile_feature_extraction.pytorch_models.supervised.models import create_model
from tactile_feature_extraction.utils.utils_learning import load_json_obj
from tactile_feature_extraction.utils.utils_learning import FTPoseEncoder
from tactile_feature_extraction.utils.utils_learning import make_save_dir_str
from tactile_feature_extraction.model_learning.setup_learning import parse_args
from tactile_feature_extraction.model_learning.setup_learning import setup_task

def make_sensor(source, working_ports):
  
    # define camera port
    if source not in working_ports:
        print(f'Camera port {source} not in working_ports: {working_ports}')
        exit()

    sensor = SimpleSensor(
        source=source,
        auto_exposure_mode=1,
        exposure=312.5,
        brightness=64,
    )

    return sensor

def process_contact_img(img):
    return process_image(
            img,
            gray=True,
            blur=15,
            thresh=[55, -2],
            dims=[240,135]
        )

class MultiDigitac(object):
    def __init__(self, sensor_source_ids=[], sensor_types=[]):

        # Define sensors 
        self.source_ids = sensor_source_ids
        self.num_sensors = len(self.source_ids)
    
        # Data buffers
        self.ft_streams = []
        self.predictions_dicts = [None] * self.num_sensors
        self.ssims = [None] * self.num_sensors
        self.contacts  = [None] * self.num_sensors
        self.raw_images = [None] * self.num_sensors
        self.contact_detect_imgs = [None] * self.num_sensors
        self.fps = []

        # create the sensor
        available_ports, working_ports, non_working_ports = list_camera_sources()
        print(f'Available Ports: {available_ports}')
        print(f'Working Ports: {working_ports}')
        print(f'Non-Working Ports: {non_working_ports}')
        self.sensors = []
        for source in self.source_ids:
            sensor = make_sensor(source, working_ports)
            self.sensors.append(sensor)

        # Get master images for ssim 
        self.master_imgs = []
        for sensor in self.sensors:
            master_img = sensor.get_image()
            master_img = process_contact_img(
                    master_img,
                )
            self.master_imgs.append(master_img)

        # create the models
        self.args = parse_args()
        self.async_data = self.args.async_data
        task = "linshear_surface_3d"
        model_type = "simple_cnn"
        self.device = self.args.device
        print(f'device = {self.device}')

        self.models = []
        self.label_encoders = []
        for sensor_type in sensor_types:
            # task specific parameters
            out_dim, label_names = setup_task(task)

            # set save dir
            save_dir_str = make_save_dir_str(self.async_data, task, sensor_type)
            save_dir = os.path.join(BASE_MODEL_PATH, save_dir_str, model_type)

            # setup parameters
            network_params = load_json_obj(os.path.join(save_dir, 'model_params'))

            # get the pose limits used for encoding/decoding pose/predictions
            ft_pose_params = load_json_obj(os.path.join(save_dir, 'ft_pose_params'))
            ft_pose_limits = [
                ft_pose_params['pose_llims'],
                ft_pose_params['pose_ulims'],
                ft_pose_params['ft_llims'],
                ft_pose_params['ft_ulims'],
            ]

            # TODO: Hardcoded for now whilst figuring out the best way of handling this
            # have to take into account raw_image_dataset -> processed_image_dataset -> process_image_on_loading
            self.processing_params = load_json_obj(os.path.join(save_dir, 'image_processing_params'))
            in_dim = self.processing_params['dims']
            in_channels = 1

            # create the model
            model = create_model(
                in_dim,
                in_channels,
                out_dim,
                network_params,
                saved_model_dir=save_dir,  # loads weights of best_model.pth
                device=self.device
                #device='cpu'
            )
            model.eval()
            self.pytorch_models.append(model)

            label_encoder = FTPoseEncoder(label_names, ft_pose_limits, self.device)
            self.label_encoders.append(label_encoder)

        self.ft_stream_run = False
        self.program_stop = False
        
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('exiting...')
        self.close()

    def start(self):
        if not self.ft_streams:
            for idx in range(self.num_sensors):
                ft_stream = Thread(None, self.run_inference_loop, args=(idx,))
                self.ft_streams.append(ft_stream)
                print('Starting stream, ', idx)

            self.ft_stream_run = True
            
            for stream in self.ft_streams:
                stream.start()

            print('prediction loop running')

    def stop(self):
        # Terminates resources if they are running
        self.ft_stream_run = False  
        
        # Joins image stream thread
        for stream in self.ft_streams:
            stream.join()    

        print('thread joined successfully')
        
        cv2.destroyAllWindows()
        print('CV windows closed.')

    def close(self):
        # Release resources and clean up ...
        self.stop()

    def run_inference_loop(self, idx):
        
        while self.ft_stream_run:
            t0 = time.time()
            # get raw image
            self.raw_images[idx] = self.sensors[idx].get_image()

            # process image using same params as training
            processed_image = process_image(
                self.raw_images[idx],
                gray=True,
                **self.processing_params
            )
            # process non-normalised image for SSIM contact detection
            self.contact_detect_imgs[idx] = process_contact_img(
                self.raw_images[idx],
            )
            
            # Get ssim and contact information
            self.ssims[idx] = ssim(np.squeeze(self.master_imgs[idx]), 
                                np.squeeze(self.contact_detect_imgs[idx]), 
                                data_range=255, 
                                multichannel=False)
            
            if  self.ssims[idx] <= 0.5:
                self.contacts[idx] = 1
            else:
                self.contacts[idx] = 0

            # put the channel into first axis because pytorch
            processed_image = np.rollaxis(processed_image, 2, 0)

            # add batch dim
            processed_image = processed_image[np.newaxis, ...]

            # convert np array image to torch tensor
            model_input = Variable(torch.from_numpy(processed_image)).float().to(self.device)
            raw_predictions = self.models[idx](model_input)

            # decode the prediction
            self.predictions_dicts[idx] = self.label_encoders[idx].decode_label(raw_predictions)

            #print("\nPredictions: ", end="")
            for label_name in self.label_encoders[idx].target_label_names:
                self.predictions_dicts[idx][label_name] = self.predictions_dicts[idx][label_name].detach().cpu().numpy()
                #print(label_name, predictions_dict[label_name])

            self.fps.append(1.0 / (time.time() - t0))
        
        self.program_stop = True

    def display(self):
        for i, frame in enumerate(self.contact_detect_imgs):
            if frame is not None:
                cv2.imshow(f"Camera {self.source_ids[i]}", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.ft_stream_run = False

    def contact_detected(self):
        if any(self.ssims):
            return True
        else:
            return False

    def get_predictions(self):
        return self.predictions_dicts

    def get_contacts(self):
        return self.ssims, self.contacts
    
    def get_fps(self):
        return np.mean(np.array(self.fps))

if __name__ == '__main__':

    # sensors: [thumb, index, middle, pinky]
    sensor_source_ids = [4, 6, 8, 13]
    sensor_types = [["nanoTip"], ["nanoTip"], ["nanoTip"], ["nanoTip"]]
    with MultiDigitac(sensor_source_ids, sensor_types) as tactip:
        tactip.start()
        while not tactip.program_stop:
            tactip.display()
            if not tactip.contact_detected():
                print('waiting...')
            else:
                # print(tactip.get_predictions())
                current_ssim, current_contacts = tactip.get_contacts()
                print('ssim {} contacts {}'.format(current_ssim, current_contacts))

            print(f'FPS = {tactip.get_fps()}')

        print('program stopped')
