import os
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt

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

def run_inference_loop(sensor, model, processing_params, label_encoder, device):

    # create cv window for plotting
    display_name = "processed_image"
    cv2.namedWindow(display_name)

    # setup bar plot
    fig = plt.figure(figsize=(15, 10))
    ax1 = fig.add_subplot(1, 2, 1)
    bars = ax1.bar(label_names, np.zeros(shape=[len(label_names)]))
    ax1.set_ylim([-5, 5])

    # setup contact point plot
    tip_rad = 11.2
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    graph,  = ax2.plot([], [], [], linestyle="", marker="o", color='b')

    # setup contact force plot
    quiver_fx = ax2.quiver(0, 0, 0, 0, 0, 0)
    quiver_fy = ax2.quiver(0, 0, 0, 0, 0, 0)
    quiver_fz = ax2.quiver(0, 0, 0, 0, 0, 0)

    # plot hemisphere wireframe
    pi = np.pi
    cos = np.cos
    sin = np.sin
    phi, theta = np.mgrid[0.0:0.5*pi:180j,
                          0.0:2.0*pi:720j]  # phi = alti, theta = azi
    x_sphere = tip_rad*sin(phi)*cos(theta)
    y_sphere = tip_rad*sin(phi)*sin(theta)
    z_sphere = tip_rad*cos(phi)
    ax2.plot_wireframe(x_sphere, y_sphere, z_sphere, color="k", linewidth=0.2)
    set_3d_axes_equal(ax2)

    step = 0 
    while True:
        start_time = time.time()

        # get raw image
        raw_image = sensor.get_image()

        # process image using same params as training
        # TODO: might want to overwrite these if using a different sensor than training
        processed_image = process_image(
            raw_image,
            gray=True,
            **processing_params
        )
        display_image = processed_image.copy()

        # put the channel into first axis because pytorch
        processed_image = np.rollaxis(processed_image, 2, 0)

        # add batch dim
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

        # # plot bar graph
        for label_name, bar in zip(label_encoder.target_label_names, bars):
            if label_name in ['qz', 'Rx', 'Ry']:
                bar.set_height(predictions_dict[label_name][0]/10)
            else:
                bar.set_height(predictions_dict[label_name][0])

        # # plot 3d scatter of contact point
        point_offset = 0.5
        point = np.array([0.0, 0.0, tip_rad + point_offset])
        roll = predictions_dict['Rx'][0]
        pitch = predictions_dict['Ry'][0]

        rot_mat = cri.transforms.euler2mat(
            [0.0, 0.0, 0.0, roll, pitch, 0.0])[:3, :3]
        x, y, z = np.dot(point, rot_mat)

        graph.set_data(x, y)
        graph.set_3d_properties(z)

        # plot contact normal
        # quiver_fx.remove()
        # quiver_fy.remove()
        quiver_fz.remove()

        # fx_scale = -predictions_dict['Fx'][0] * 5.0
        # fy_scale = -predictions_dict['Fy'][0] * 5.0
        # fz_scale = -predictions_dict['Fz'][0] * 5.0

        fz_scale = -np.linalg.norm([predictions_dict['Fx'][0], predictions_dict['Fy'][0], predictions_dict['Fz'][0]]) * 5.0

        # fxu, fxv, fxw = np.dot(np.array([1.0, 0.0, 0.0]), rot_mat)
        # fyu, fyv, fyw = np.dot(np.array([0.0, 1.0, 0.0]), rot_mat)
        fzu, fzv, fzw = np.dot(np.array([0.0, 0.0, 1.0]), rot_mat)

        # quiver_fx = ax2.quiver(x, y, z, fxu, fxv, fxw,
        #                        length=fx_scale, normalize=True, color='r')
        # quiver_fy = ax2.quiver(x, y, z, fyu, fyv, fyw,
        #                        length=fy_scale, normalize=True, color='g')
        quiver_fz = ax2.quiver(x, y, z, fzu, fzv, fzw,
                               length=fz_scale, normalize=True, color='b')
        
        # # Save figure
        # root_directory = '/home/max-yang/Documents/Projects/allegro/smg_gym/smg_gym/ppo_adapt_helpers/analysis/tactile_data/tactile_images'
        # save_image_path = os.path.join(root_directory, f'frame_{step}.png')
        # plt.savefig(save_image_path)

        # Control the framerate
        plt.pause(0.001)

        cv2.imshow(display_name, display_image)
        k = cv2.waitKey(10)
        if k == 27:  # Esc key to stop
            break

        print('FPS: ', 1.0 / (time.time() - start_time))
        step += 1

if __name__ == '__main__':

    args = parse_args()
    async_data = args.async_data
    tasks = args.tasks
    sensors = args.sensors
    models = args.models
    device = args.device

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

            # if async_data:
            #     processing_params = load_json_obj(os.path.join(save_dir, 'frame_processing_params'))
            #     in_dim = processing_params['dims']
            #     in_channels = processing_params['n_stack']
            # else:
            #     processing_params = load_json_obj(os.path.join(save_dir, 'image_processing_params'))
            #     in_dim = processing_params['dims']
            #     in_channels = 1

            # TODO: Hardcoded for now whilst figuring out the best way of handling this
            # have to take into account raw_image_dataset -> processed_image_dataset -> process_image_on_loading
            processing_params = load_json_obj(os.path.join(save_dir, 'image_processing_params'))
            # processing_params = {
            #     'dims': (256, 256),
            #     'bbox': [75, 30, 525, 480],
            #     'thresh': [11, -30],
            #     'stdiz': False,
            #     'normlz': True,
            #     'circle_mask_radius': 180,
            # }
            in_dim = processing_params['dims']
            in_channels = 1

            # create the model
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
            run_inference_loop(sensor, model, processing_params, label_encoder, device)
