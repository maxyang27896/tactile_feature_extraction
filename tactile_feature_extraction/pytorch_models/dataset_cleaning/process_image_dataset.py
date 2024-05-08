import os
from glob import glob
import pathlib
import cv2

from tactile_feature_extraction.utils.image_transforms import process_image
from tactile_feature_extraction.utils.utils_learning import make_dir, save_json_obj

data_path = os.path.join(
    # "/home/alex/tactile_datasets/tactile_servo_control/tactip_127"
    # "/home/alex/tactile_datasets/braille_classification/tactip_331_25mm"
    "/home/alex/tactile_datasets/tactile_sim2real"
)


def process_dataset(
    base_dir,
    image_processing_params,
    dry_run=True
):

    image_dir = os.path.join(
        base_dir,
        'images'
    )

    all_image_files = [y for x in os.walk(
        image_dir
    ) for y in glob(os.path.join(x[0], '*.png'))]

    cv2.namedWindow("proccessed_image")

    # make new dir for saving images
    new_dir_name = os.path.join(
        base_dir,
        f"processed_images_{image_processing_params['dims'][0]}",
    )

    if not dry_run:
        make_dir(new_dir_name)

    for image_file in all_image_files:

        # process image
        raw_image = cv2.imread(image_file)

        # preprocess/augment image
        processed_image = process_image(
            raw_image,
            gray=True,
            **image_processing_params
        )

        # create new filename for saving
        image_path = pathlib.Path(image_file)
        filename = image_path.stem
        new_image_filename = os.path.join(
            new_dir_name,
            filename + '.png'
        )
        print(new_image_filename)

        # save the new image
        if not dry_run:
            cv2.imwrite(new_image_filename, processed_image)

        # show image
        cv2.imshow("proccessed_image", processed_image)
        k = cv2.waitKey(1)
        if k == 27:    # Esc key to stop
            exit()


if __name__ == '__main__':

    image_processing_params = {
        'dims': (64, 64),
        'bbox': [80, 25, 530, 475],
        'thresh': [11, -30],
        'stdiz': False,
        'normlz': False,
        'circle_mask_radius': 220,
    }

    dry_run = False

    # tasks = ['edge_2d', 'edge_3d', 'edge_5d', 'surface_3d']
    # tasks = ['alphabet', 'arrows']
    # tasks = ['surface_3d']
    # tasks = ['edge_2d', 'surface_3d']
    tasks = ['spherical_probe']
    collection_modes = ['tap']
    sets = ['train', 'val']

    for task in tasks:
        for collection_mode in collection_modes:
            for set in sets:

                base_dir = os.path.join(
                    data_path,
                    task,
                    'tactip_331',
                    collection_mode,
                    set,
                )

                if not dry_run:
                    save_json_obj(image_processing_params, os.path.join(base_dir, 'image_processing_params'))

                process_dataset(
                    base_dir,
                    image_processing_params=image_processing_params,
                    dry_run=dry_run
                )
