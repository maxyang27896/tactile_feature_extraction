# just the essentials, for control

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

class TacTip(object):
    def __init__(self, sensor_id):
        self.ft_stream = None
        self.ft_stream_run = False
        self.predictions_dict = None
        self.raw_image = None
        self.contact_detect_img = None

        self.args = parse_args()
        self.async_data = self.args.async_data
        self.tasks = self.args.tasks
        self.sensors = self.args.sensors
        self.models = self.args.models
        self.device = self.args.device
        print(f'device = {self.device}')

        self.fps = []

        # create the sensor
        self.sensor = self.make_sensor(source=sensor_id)
        self.program_stop = False
        
        for task in self.tasks:
            for model_type in self.models:

                # task specific parameters
                out_dim, label_names = setup_task(task)

                # set save dir
                save_dir_str = make_save_dir_str(self.async_data, task, self.sensors)
                save_dir = os.path.join(BASE_MODEL_PATH, save_dir_str, model_type)

                # setup parameters
                network_params = load_json_obj(os.path.join(save_dir, 'model_params'))
                learning_params = load_json_obj(os.path.join(save_dir, 'learning_params'))

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
                self.model = create_model(
                    in_dim,
                    in_channels,
                    out_dim,
                    network_params,
                    saved_model_dir=save_dir,  # loads weights of best_model.pth
                    device=self.device
                    #device='cpu'
                )
                self.model.eval()

                self.label_encoder = FTPoseEncoder(label_names, ft_pose_limits, self.device)

        self.master_img = self.sensor.get_image()
        self.master_img = process_image(
                self.master_img,
                gray=True,
                dims=[240,135]
            )
        
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('exiting...')
        self.close()

    def start(self):
        if not self.ft_stream:
            self.ft_stream = Thread(None, self.run_inference_loop)
            # self.ssim_stream = Thread(None, self.contact_detect)
            self.ft_stream_run = True
            self.ft_stream.start()
            # self.ssim_stream.start()
            print('prediction loop running')

    def make_sensor(self, source=0):
        available_ports, working_ports, non_working_ports = list_camera_sources()

        print(f'Available Ports: {available_ports}')
        print(f'Working Ports: {working_ports}')
        print(f'Non-Working Ports: {non_working_ports}')
        
        # source = 0 # define camera port
        if source not in working_ports:
            print(f'Camera port {source} not in working_ports: {working_ports}')
            exit()

        sensor = SimpleSensor(
            source=source,
            auto_exposure_mode=1,
            exposure=312.5
        )

        return sensor
    
    def get_predictions(self):
        return self.predictions_dict
    
    def similarity(self, img):
        # SSIM CALCULATOR #
        similarity = ssim(np.squeeze(self.master_img), np.squeeze(img), data_range=255, multichannel=False)
        return similarity

    def contact_detect(self):
        if self.contact_detect_img is not None:
            similarity = self.similarity(self.contact_detect_img)
            if  similarity <= 0.9:
                contact = True
            else:
                contact = False
            return similarity, contact
        else:
            return None
    
    def get_fps(self):
        return np.mean(np.array(self.fps))

    def run_inference_loop(self):
        
        while self.ft_stream_run:
            t0 = time.time()

            # get raw image
            self.raw_image = self.sensor.get_image()

            # process image using same params as training
            processed_image = process_image(
                self.raw_image,
                gray=True,
                **self.processing_params
            )
            # process non-normalised image for SSIM contact detection
            self.contact_detect_img = process_image(
                self.raw_image,
                gray=True,
                dims=[240,135]
            )

            # put the channel into first axis because pytorch
            processed_image = np.rollaxis(processed_image, 2, 0)

            # add batch dim
            processed_image = processed_image[np.newaxis, ...]

            # convert np array image to torch tensor
            model_input = Variable(torch.from_numpy(processed_image)).float().to(self.device)
            raw_predictions = self.model(model_input)

            # decode the prediction
            self.predictions_dict = self.label_encoder.decode_label(raw_predictions)

            #print("\nPredictions: ", end="")
            for label_name in self.label_encoder.target_label_names:
                self.predictions_dict[label_name] = self.predictions_dict[label_name].detach().cpu().numpy()
                #print(label_name, predictions_dict[label_name])

            #cv2.imshow('master_img', self.master_img)
            # cv2.imshow('current_img', self.contact_detect_img)
            # k = cv2.waitKey(10)
            # if k == 27:  # Esc key to stop
            #     cv2.destroyAllWindows()
            #     break

            self.fps.append(1.0 / (time.time() - t0))
        
        self.program_stop = True

    def display(self):
        if self.contact_detect_img is not None:
            cv2.imshow(f"Camera img", self.contact_detect_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.ft_stream_run = False

    def stop(self):
        # Terminates resources if they are running
        if self.ft_stream_run:
            self.ft_stream_run = False   # Breaks image stream main loop
            self.ft_stream.join()    # Joins image stream thread
            # self.ssim_stream.join()     # Joins contact detection thread
            cv2.destroyAllWindows()
            print('thread joined successfully')

    def close(self):
        # Release resources and clean up ...
        self.stop()

if __name__ == '__main__':

    sensor_id = 2
    with TacTip(sensor_id) as tactip:
        tactip.start()
        while not tactip.program_stop:
            tactip.display()
            if not tactip.contact_detect():
                print('waiting...')
            elif tactip.contact_detect()[1]==True:
                # print(tactip.get_predictions())
                print('contact detected, ssim', tactip.contact_detect()[0])
            else:
                print(f'no contact detected, ssim = {tactip.contact_detect()[0]}')

            print(f'FPS = {tactip.get_fps()}')
