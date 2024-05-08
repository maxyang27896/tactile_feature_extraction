import os
import pathlib
import pandas as pd

data_path = os.path.join(
    "/home/alex/tactile_datasets/braille_classification/tactip_331_25mm"
)

if __name__ == '__main__':

    tasks = ['alphabet', 'arrows']
    sets = ['train', 'val']

    dry_run = True

    for task in tasks:
        for set in sets:

            base_dir = os.path.join(
                data_path,
                task,
                set,
            )

            def adjust_filename(video_filename):
                video_filename = pathlib.Path(video_filename).stem
                id = video_filename.split('_')[1]
                return f'image_{id}.png'

            target_df = pd.read_csv(os.path.join(base_dir, 'targets_video.csv'))
            target_df['sensor_image'] = target_df.sensor_video.apply(adjust_filename)
            target_df.drop('sensor_video', axis=1, inplace=True)

            print(target_df)

            if not dry_run:
                target_df.to_csv(os.path.join(base_dir, 'targets.csv'), index=False)
