import os
import itertools

from tactile_feature_extraction.model_learning.setup_learning import parse_args
from tactile_feature_extraction.model_learning.setup_learning import setup_model
from tactile_feature_extraction.model_learning.setup_learning import setup_learning
from tactile_feature_extraction.model_learning.setup_learning import setup_task
from tactile_feature_extraction.model_learning.evaluate_model import evaluate_model

from tactile_feature_extraction.pytorch_models.supervised.models import create_model
from tactile_feature_extraction.pytorch_models.supervised.train_model_w_metrics import train_model_w_metrics
from tactile_feature_extraction.pytorch_models.supervised.image_generator import ImageDataGenerator
from tactile_feature_extraction.pytorch_models.supervised.frame_generator import FrameDataGenerator

from tactile_feature_extraction.utils.utils_plots import ErrorPlotter
from tactile_feature_extraction.utils.utils_learning import FTPoseEncoder
from tactile_feature_extraction.utils.utils_learning import seed_everything, make_dir
from tactile_feature_extraction.utils.utils_learning import get_ft_pose_limits
from tactile_feature_extraction.utils.utils_learning import csv_row_to_label
from tactile_feature_extraction.utils.utils_learning import make_save_dir_str

from tactile_feature_extraction import BASE_DATA_PATH
from tactile_feature_extraction import BASE_MODEL_PATH
from tactile_feature_extraction import SAVED_MODEL_PATH


def launch():

    args = parse_args()
    async_data = args.async_data
    tasks = args.tasks
    sensors = args.sensors
    models = args.models
    device = args.device

    for task in tasks:
        for model_type in models:

            save_dir_str = make_save_dir_str(async_data, task, sensors)

            # task specific parameters
            out_dim, label_names = setup_task(task)

            # setup save dir
            save_dir = os.path.join(BASE_MODEL_PATH, save_dir_str, model_type)
            make_dir(save_dir)

            # setup parameters
            model_params = setup_model(model_type, save_dir)
            learning_params, image_processing_params, frame_processing_params, augmentation_params = setup_learning(save_dir)

            if async_data:
                DataGenerator = FrameDataGenerator
                train_processing_params = {**frame_processing_params, **augmentation_params}
                val_processing_params = frame_processing_params
                in_dim = frame_processing_params['dims']
                in_channels = frame_processing_params['n_stack']
            else:
                DataGenerator = ImageDataGenerator
                train_processing_params = {**image_processing_params, **augmentation_params}
                val_processing_params = image_processing_params
                in_dim = image_processing_params['dims']
                in_channels = 1

            # create the model
            seed_everything(learning_params['seed'])
            model = create_model(
                in_dim=in_dim,
                in_channels=in_channels,
                out_dim=out_dim,
                model_params=model_params,
                device=device,
                # saved_model_dir=SAVED_MODEL_PATH                  # Uncomment to allow for transfer learning
            )

            combined_dirs = list(itertools.product([task], sensors))
            combined_paths = [os.path.join(*i) for i in combined_dirs]

            train_data_dirs = [
                os.path.join(BASE_DATA_PATH, data_path, "train")
                for data_path in combined_paths
            ]
            ft_pose_limits = get_ft_pose_limits(train_data_dirs, save_dir)

            val_data_dirs = [
                os.path.join(BASE_DATA_PATH, data_path, "val")
                for data_path in combined_paths
            ]

            # set generators and loaders
            train_generator = DataGenerator(
                data_dirs=train_data_dirs,
                csv_row_to_label=csv_row_to_label,
                **train_processing_params
            )
            val_generator = DataGenerator(
                data_dirs=val_data_dirs,
                csv_row_to_label=csv_row_to_label,
                **val_processing_params
            )

            # create the encoder/decoder for labels
            label_encoder = FTPoseEncoder(label_names, ft_pose_limits, device)

            # create instance for plotting errors
            error_plotter = ErrorPlotter(
                target_label_names=label_names,
                save_dir=save_dir,
                name='error_plot.png',
                plot_during_training=False
            )

            # Train model
            train_model_w_metrics(
                model,
                label_encoder,
                train_generator,
                val_generator,
                learning_params,
                save_dir,
                error_plotter=error_plotter,
                calculate_train_metrics=False,
                device=device
            )

            # perform a final evaluation using the best modely
            evaluate_model(
                task,
                model,
                label_encoder,
                val_generator,
                learning_params,
                save_dir,
                error_plotter,
                device=device
            )


if __name__ == "__main__":
    launch()
