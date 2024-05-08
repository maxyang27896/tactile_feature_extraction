import os
import pandas as pd
import numpy as np
import math
import shutil
import time

from ipdb import set_trace

from FiltDataGather import DataGather
from sample_analysis import Analyser
from cri.robot import SyncRobot, AsyncRobot
from cri.controller import RTDEController

from tactile_feature_extraction import TIP_ID
from tactile_feature_extraction import ROOT_PATH
from tactile_feature_extraction import BASE_DATA_PATH

def make_target_df(target_df_file, poses_rng, num_poses, num_frames=1, obj_poses=[[0,]*6], moves_rng=[[0,]*6,]*2, **kwargs):
    np.random.seed(0) # make predictable
    poses = np.random.uniform(low=poses_rng[0], high=poses_rng[1], size=(num_poses, 6))
    poses = poses[np.lexsort((poses[:,1], poses[:,5]))]
    moves = np.random.uniform(low=moves_rng[0], high=moves_rng[1], size=(num_poses, 6))

    # Randomly allow a small amount of non contacted data 
    # for pose in poses:
    #     if np.random.uniform(0, 1) < 0.05:
    #         pose[2] = np.random.uniform(0, 0.99)
    #         pose[3] = 0.0
    #         pose[4] = 0.0

    # Filter out shear moves when poses_2 (depth) is below 2mm
    # indices = poses[:, 2]  < 2.0
    # moves[indices, 0] = 0.0
    # moves[indices, 1] = 0.0

    pose_ = [f"pose_{_+1}" for _ in range(6)]
    move_ = [f"move_{_+1}" for _ in range(6)]
    target_df = pd.DataFrame(columns=["image_name", "data_name", "obj_id", "pose_id", *pose_, *move_])
    for i in range(num_poses * len(obj_poses)):
        data_name = f"frame_{i}"
        i_pose, i_obj = (int(i%num_poses), int(i/num_poses))
        pose = poses[i_pose,:] + obj_poses[i_obj]
        move = moves[i_pose,:]        
        for f in range(num_frames):
            frame_name = f"frame_{i}_{f}.png"
            target_df.loc[-1] = np.hstack((frame_name, data_name, i_obj+1, i_pose+1, pose, move))
            target_df.index += 1

    target_df.to_csv(target_df_file, index=False)
    return target_df

def collect(target_df, dataPath, resume_from, sleep_time, i):
    
    # collect one data and flush the sample
    flush = False
    for _, row in target_df.iloc[resume_from:].iterrows():
        if not flush:
            # Take first sample and disregard to clear buffer:
            pose = row.loc['pose_1' : 'pose_6'].values.astype(float)
            move = row.loc['move_1' : 'move_6'].values.astype(float)
            print(f'pose for frame_{i} = {pose}')
            # Add extra depth to account additional geomrtry in Rx direction
            add_tap_depth = max((22 - abs(pose[3]))/22 * 1.5, 0.0)
            add_tap = [0,0,add_tap_depth,0,0,0]
            # print('Tap deep', add_tap_depth)
            tap = [0,0,pose[2] ,0,0,0]
            pose = (pose - tap)
            robot.move_linear(pose - move)
            dg.begin_sample(i)
            robot.move_linear(pose - move + tap + add_tap)
            robot.linear_speed = 10
            time.sleep(0.25)
            robot.move_linear(pose + tap + add_tap)
            time.sleep(sleep_time)
            dg.stop_and_write()
            robot.linear_speed = 200
            robot.move_linear((0, 0, -10, 0, 0, 0))
            time.sleep(sleep_time)
            os.remove(f'{dataPath}/time_series/sample_{i}.pkl')
            shutil.rmtree(f'{dataPath}/videos/sample_{i}')
            flush = True
        else:
            print('flushed - moving on to main samples')
            break
    
    # Main data collection
    for _, row in target_df.iloc[resume_from:].iterrows():
        try:
            # Get pose:
            i_obj, i_pose = (int(row.loc["obj_id"]), int(row.loc["pose_id"]))
            pose = row.loc['pose_1' : 'pose_6'].values.astype(float)
            move = row.loc['move_1' : 'move_6'].values.astype(float)
            print(f'pose for frame_{i} = {pose}')
            # Add extra depth to account additional geomrtry in Rx direction
            add_tap_depth = max((22 - abs(pose[3]))/22 * 1.5, 0.0)
            add_tap = [0,0,add_tap_depth,0,0,0]
            # print('Tap deep', add_tap_depth)
            tap = [0,0,pose[2] ,0,0,0]
            pose = (pose - tap)
            robot.move_linear(pose - move)
            dg.begin_sample(i)
            robot.move_linear(pose - move + tap + add_tap)
            robot.linear_speed = 10
            time.sleep(0.25)
            robot.move_linear(pose + tap + add_tap)
            time.sleep(sleep_time)
            dg.stop_and_write()
            robot.linear_speed = 200
            robot.move_linear((0, 0, -10, 0, 0, 0))
            # time.sleep(sleep_time)

            sample_size = os.path.getsize(f'{dataPath}/time_series/sample_{i}.pkl') #check FT sensor is still working

            if sample_size < 55000:
                dg.pause()
                os.remove(f'{dataPath}/time_series/sample_{i}.pkl')
                shutil.rmtree(f'{dataPath}/videos/sample_{i}')
                print(f'sample {i} under threshold at {sample_size}, removed and exiting...')
                break

            i = i+1   
        except:
            print(f'something went wrong sample_{i} - moving on...')
            break

tcp = 65
offset = 0                # For tips 0 degrees
# offset = 6                  # For tips 45 and 90 degrees
base_frame = (0, 0, 0, 0, 0, 0)  
# base frame: x->front, y->right, z->up (higher z to make sure doesnt press into the table)
work_frame = (473, -111, 60.75-offset, -180, 0, -90)            #  0 degrees
# work_frame = (473, -111, 66.75-offset, -180, 0, -90)            # For tips 45 and 90 degrees
# work_frame = (473, -40, 61-offset, -180, 0, -90)           # safe baseframe for testing, using a box
tcp_x_offset = -1.5                 # 0 degrees
# tcp_x_offset = -1.75                 # For tips 45 degrees
tcp_y_offset = 1.5
tcp_x = tcp_x_offset*math.sin(math.pi/4) + tcp_y_offset*math.cos(math.pi/4)
tcp_y = tcp_x_offset*math.cos(math.pi/4) - tcp_y_offset*math.sin(math.pi/4)

# Resume from last completed sample +1 (0 for new dataset):
resume_from = 0

if resume_from == 0:
    resume = False
    poses_rng = [[0, 0, 1.0, 28.0, 28.0, 0], [0, 0, 4.0, -28.0, -28.0, 0]]    # pose ranges (min values, max values)
    num_poses = 3000
    num_frames = 1
    moves_rng = [[2, 2, 0, 0, 0, 0], [-2, -2, 0, 0, 0, 0]]    # Shear movements (min values, max values)
    
    # Make data path
    folder = f"collect_{TIP_ID}_5D_surface" 
    dataPath = os.path.join(ROOT_PATH, folder)
    os.makedirs(dataPath, exist_ok=True)

    target_df = make_target_df(f"{dataPath}/targets.csv", poses_rng, num_poses, num_frames, obj_poses=[[0,]*6], moves_rng=moves_rng)
else:
    resume = True
    dataPath = BASE_DATA_PATH
    target_df = pd.read_csv(f'{dataPath}/targets.csv')

with DataGather(resume=resume, dataPath=dataPath) as dg, AsyncRobot(SyncRobot(RTDEController(ip='192.11.72.20'))) as robot:

    time.sleep(1)
    analyse = Analyser(TIP_ID)

    # Setup robot (TCP, linear speed,  angular speed and coordinate frame):
    robot.tcp = (tcp_x, tcp_y, tcp + offset-0.25, 0, 0, -45) 
    robot.axes = "sxyz"
    robot.linear_speed = 100
    robot.angular_speed = 100
    robot.coord_frame = work_frame
    sleep_time = 0.5
    angle = 0
    i = resume_from
    flush = False
    robot.move_linear((0, 0, 0, 0, 0, 0)) #move to home position
    print('Moved to home position')
    set_trace()
    
    # Main data collection
    time.sleep(3)
    dg.start()
    time.sleep(2)
    collect(target_df, dataPath, resume_from, sleep_time, i)
    dg.stop()

    robot.linear_speed = 30
    robot.move_linear((0, 0, -50, 0, 0, 0)) #move to a bit higher position to avoid damaging the sensor