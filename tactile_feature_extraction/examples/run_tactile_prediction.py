import cv2
import os
import time
import shutil
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import torch
from torch.autograd import Variable

import cri.transforms

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


def make_sensor(source=0):
    available_ports, working_ports, non_working_ports = list_camera_sources()

    print(f'Available Ports: {available_ports}')
    print(f'Working Ports: {working_ports}')
    print(f'Non-Working Ports: {non_working_ports}')

    if source not in working_ports:
        print(f'Camera port {source} not in working_ports: {working_ports}')
        exit()

    sensor = SimpleSensor(
        source=source,
        auto_exposure_mode=1,
        exposure=312.5,
        brightness=64
    )

    return sensor


def set_3d_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

def plot_contact_patch(ax, x, y, z, sphere_radius, patch_radius, theta_center, azimuth_center, color):

    # Calculate the cartesian coordinates of the patch center
    x_center = sphere_radius * np.sin(theta_center) * np.cos(azimuth_center)
    y_center = sphere_radius * np.sin(theta_center) * np.sin(azimuth_center)
    z_center = sphere_radius * np.cos(theta_center)

    distances = np.sqrt((x - x_center)**2 + (y - y_center)**2 + (z - z_center)**2)
    patch = distances <= patch_radius

    # Check which points are within the patch
    distances = np.sqrt((x - x_center)**2 + (y - y_center)**2 + (z - z_center)**2)
    patch = distances <= patch_radius

    # Highlight the patch
    scatter = ax.scatter(x[patch], y[patch], z[patch], color=color, alpha=0.01)
    return scatter

def convert_to_video(image_folder, out_file, fps):

    # Get image size
    images = [img for img in os.listdir(image_folder)]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width = frame.shape[:2]

    # Save video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(out_file, fourcc, fps, (width, height))
    for i in range(len(images)):
        video_writer.write(cv2.imread(os.path.join(image_folder, f'frame_{i}.png')))
    video_writer.release()

def process_contact_img(img):
    return process_image(
            img,
            gray=True,
            blur=15,
            thresh=[55, -2],
            dims=[240, 135]
        )

def save_prediction_plots(df, output_dir):

    # Create plot figure
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(1, 1, 1, projection='3d')

    # create hemisphere wireframe
    tip_rad = 11.2
    pi = np.pi
    cos = np.cos
    sin = np.sin
    phi, theta = np.mgrid[0.0:0.5*pi:180j,
                        0.0:2.0*pi:720j]  # phi = alti, theta = azi
    x_sphere = tip_rad*sin(phi)*cos(theta)
    y_sphere = tip_rad*sin(phi)*sin(theta)
    z_sphere = tip_rad*cos(phi)
    ax.plot_wireframe(x_sphere, y_sphere, z_sphere, color="k", linewidth=0.2)
    set_3d_axes_equal(ax)
    ax.grid(False)
    ax.set_axis_off()
    # plt.xlabel('Contact Pose Y')
    # plt.ylabel('Contact Pose X')
    plt.tick_params(left = False, right = False , labelleft = False , 
                    labelbottom = False, bottom = False) 
    plt.tight_layout()

    # Loop through and plot contact patches
    point_offset = 0.0
    point = np.array([0.0, 0.0, tip_rad + point_offset])
    for i in range(len(df)):    
        # Get data
        roll = df.iloc[i]['roll']
        pitch = df.iloc[i]['pitch']
        force = df.iloc[i]['force']
        fz_scale = force * 2.0

        # Get contact location
        rot_mat = cri.transforms.euler2mat(
            [0.0, 0.0, 0.0, roll, pitch, 0.0])[:3, :3]
        x, y, z = np.dot(point, rot_mat)
        
        # Plot contact region
        phi0 = np.arctan2(y, x)  
        theta0 = np.arccos(z / tip_rad)
        scatter = plot_contact_patch(ax,  
                                x_sphere, 
                                y_sphere, 
                                z_sphere,
                                sphere_radius=tip_rad, 
                                patch_radius=fz_scale, 
                                theta_center=theta0, 
                                azimuth_center=phi0,
                                color='r')
        
        plt.savefig(os.path.join(output_dir, f'frame_{i}.png'))
        scatter.remove()

def run_inference_loop(sensor, model, processing_params, label_encoder, device, save_video = False):

    desired_hz = 20.0  # Target frequency in Hz
    period = 1.0 / desired_hz  # Time per loop iteration in seconds

    # create cv window for plotting
    display_name = "processed_image"
    cv2.namedWindow(display_name)

    # Define directories
    root_data_dir = '/home/max-yang/Documents/Projects/allegro/smg_gym/smg_gym/ppo_adapt_helpers/analysis/tactile_data'
    tactile_data_dir = os.path.join(root_data_dir, 'prediction_data.csv')
    tactile_img_dir = os.path.join(root_data_dir, 'tactile_images')
    tactile_plots_dir = os.path.join(root_data_dir, 'prediction_plots')
    output_tactile_video = os.path.join(root_data_dir, 'tactile_video.mp4')
    output_tactile_prediction_video = os.path.join(root_data_dir, 'tactile_prediction_video.mp4')
    df = pd.DataFrame(columns=['roll', 'pitch', 'force'])

    # Delete previous data in directories
    for dir in [tactile_img_dir, tactile_plots_dir]:
        for filename in os.listdir(dir):
            file_path = os.path.join(dir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))

    # Run loop
    step = 0 
    while True:
        start_time = time.time()

        # get raw image
        raw_image = sensor.get_image()

        # process image using same params as training
        processed_image = process_image(
            raw_image,
            gray=True,
            **processing_params
        )
        display_image = processed_image.copy()
        processed_image = np.rollaxis(processed_image, 2, 0)
        processed_image = processed_image[np.newaxis, ...]

        # convert np array image to torch tensor
        model_input = Variable(torch.from_numpy(processed_image)).float().to(device)
        raw_predictions = model(model_input)

        # decode the prediction
        predictions_dict = label_encoder.decode_label(raw_predictions)

        print("\nPredictions: ", end="\nq")
        for label_name in label_encoder.target_label_names:
            predictions_dict[label_name] = predictions_dict[label_name].detach().cpu().numpy()
            print(label_name, predictions_dict[label_name])
        roll = predictions_dict['Rx'][0]
        pitch = predictions_dict['Ry'][0]
        force = np.linalg.norm([predictions_dict['Fx'][0], predictions_dict['Fy'][0], predictions_dict['Fz'][0]])

        cv2.imshow(display_name, display_image)
        k = cv2.waitKey(10)
        if k == 27:  # Esc key to stop
            break
        
        # Save data
        df.loc[step] = [roll, pitch, force]
        cv2.imwrite(os.path.join(tactile_img_dir, f'frame_{step}.png'), (display_image*255).astype(np.uint8))

        # Control frame rate
        elapsed_time = time.time() - start_time
        sleep_time = period - elapsed_time
        if sleep_time > 0:
            time.sleep(sleep_time)
        else:
            print("Warning: Loop iteration took longer than expected")
        print('FPS: ', 1.0 / (time.time() - start_time))

        step += 1

    # save data frame
    print(f'Saving Data to {tactile_data_dir}')
    df.to_csv(tactile_data_dir, index=False)
    
    # Save tactile video
    print(f'Saving video to {output_tactile_video}...')
    convert_to_video(tactile_img_dir, output_tactile_video, desired_hz)

    # setup contact point plot
    print(f'Creating prediction plots and saving to {tactile_plots_dir}')
    save_prediction_plots(df, tactile_plots_dir)
    print(f'Saving prediction video to {output_tactile_prediction_video}...')
    convert_to_video(tactile_plots_dir, output_tactile_prediction_video, desired_hz)

if __name__ == '__main__':

    args = parse_args()
    async_data = args.async_data
    tasks = args.tasks
    sensors = args.sensors
    models = args.models
    device = args.device
    save_video = False

    # create the sensor
    sensor = make_sensor(source=0)

    for task in tasks:
        for model_type in models:

            # task specific parameters
            out_dim, label_names = setup_task(task)

            # set save dir
            save_dir_str = make_save_dir_str(async_data, task, sensors)
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

            # create the model
            processing_params = load_json_obj(os.path.join(save_dir, 'image_processing_params'))
            in_dim = processing_params['dims']
            in_channels = 1
            model = create_model(
                in_dim,
                in_channels,
                out_dim,
                network_params,
                saved_model_dir=save_dir,  # loads weights of best_model.pth
                device=device
            )
            model.eval()

            label_encoder = FTPoseEncoder(label_names, ft_pose_limits, device)

            # start inference on live camera feed
            run_inference_loop(sensor, model, processing_params, label_encoder, device, save_video)
