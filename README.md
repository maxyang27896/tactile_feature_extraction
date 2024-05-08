# Tactile Feature Extraction: Code for tactile data collection and supervised learning

This repo contains code for tactile data collection with a vision-based tactile sensor and supervised learnig of tactile features. The data is collected using a UR5 and a force torque sensor to obtain labels for each corresponding tactile image. The data is then used for supervised learning to provide a model for predicting tactile features from tactile images in real-time. 


## Installation 
This repository require the [Common Robot Interface](https://github.com/dexterousrobot/common_robot_interface) and [Video Stream Processor](https://github.com/dexterousrobot/video_stream_processor.git) repositories to be pre-installed. Please follow the instructions within these repos first. 

```
git clone 
pip install -e .
```

## Data Collection
The data will be saved under the names and directories provided in __init__.py file by default. Please change this variables if needed. The files for data collection are under the `/data_collection` folder. 

1. To collect data, run `collect_surface_5D_FT.py`
2. After data has been collected, run `extract_data.py` to extract images and force data. 
3. Run `split_data.py` to create train and validation datasets
4. Run `process_data.py` to reformat data to PyTorch formats. 

## Model Training

The code for model learning are under the `/model_learning` directory. To train the CNN model for tactile feature prediction, run `launch_training.py`. This will save the trained model under the directories specified in the __init__.py file. 

## Inference
Example inference codes has been provided under the `/examples` folder. To visualize the predictions of the tactile sensor in real time, run `demo_realtime_predictions.py`.