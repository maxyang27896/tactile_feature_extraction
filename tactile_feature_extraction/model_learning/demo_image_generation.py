import os
import itertools

from tactile_feature_extraction import BASE_DATA_PATH
from tactile_feature_extraction.model_learning.setup_learning import parse_args
from tactile_feature_extraction.model_learning.setup_learning import setup_learning

from tactile_feature_extraction.utils.utils_learning import csv_row_to_label
from tactile_feature_extraction.utils.utils_learning import make_save_dir_str

from tactile_feature_extraction.pytorch_models.supervised.image_generator import demo_image_generation
from tactile_feature_extraction.pytorch_models.supervised.frame_generator import demo_frame_generation

if __name__ == '__main__':

    args = parse_args()
    async_data = args.async_data
    tasks = args.tasks
    sensors = args.sensors

    learning_params, image_processing_params, frame_processing_params, augmentation_params = setup_learning()

    for task in tasks:
        save_dir_str = make_save_dir_str(async_data, task, sensors)

        combined_dirs = list(itertools.product([task], sensors))
        combined_paths = [os.path.join(*i) for i in combined_dirs]

        data_dirs = [
            os.path.join(BASE_DATA_PATH, data_path, "train")
            for data_path in combined_paths
        ]

        if async_data:
            demo_frame_generation(
                data_dirs,
                csv_row_to_label,
                learning_params,
                frame_processing_params,
                augmentation_params
            )
        else:
            demo_image_generation(
                data_dirs,
                csv_row_to_label,
                learning_params,
                image_processing_params,
                augmentation_params
            )
