import os
from glob import glob
import pathlib
import cv2

from tactile_feature_extraction.utils.utils_image_processing import load_video_frames
from tactile_feature_extraction.utils.image_transforms import process_image
from tactile_feature_extraction.utils.utils_learning import make_dir

data_path = os.path.join(
    "/home/alex/tactile_datasets/braille_classification/tactip_331_25mm"
)


def process_dataset(
    base_dir,
    image_processing_params,
    dry_run=True
):

    video_dir = os.path.join(
        base_dir,
        'videos'
    )

    all_video_files = [y for x in os.walk(
        video_dir
    ) for y in glob(os.path.join(x[0], '*.mp4'))]

    cv2.namedWindow("proccessed_image")

    # make new dir for saving images
    new_dir_name = os.path.join(
        base_dir,
        'images',
    )

    if not dry_run:
        make_dir(new_dir_name)

    for video_file in all_video_files:

        # process image
        image_arr = load_video_frames(video_file)

        # take last image as raw frame
        raw_image = image_arr[-1]

        # preprocess/augment image
        processed_image = process_image(
            raw_image,
            gray=True,
            **image_processing_params
        )

        # create new filename for saving
        video_path = pathlib.Path(video_file)
        filename = video_path.stem
        id = filename.split("_")[1]
        new_image_filename = os.path.join(
            new_dir_name,
            'image_' + id + '.png'
        )

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
        'dims': None,  # (128, 128),
        'bbox': None,  # [110, 40, 510, 440],
        'thresh': False,
        'stdiz': False,
        'normlz': False,
        'circle_mask_radius': None,
    }

    dry_run = True

    # tasks = ['edge_2d', 'edge_3d', 'edge_5d', 'surface_3d']
    tasks = ['alphabet', 'arrows']
    sets = ['train', 'val']

    for task in tasks:
        for set in sets:

            base_dir = os.path.join(
                data_path,
                task,
                set,
            )

            process_dataset(
                base_dir,
                image_processing_params=image_processing_params,
                dry_run=dry_run
            )
