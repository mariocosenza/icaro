import os
import pandas as pd
from pose_landmark import pose_point
from mediapipe.tasks import python

def build_dataset_landmarks(dir: str, base_path:str):
    frame_list = []
    path = dir + base_path + '/' + base_path + '/'
    len_list = len(os.listdir(path + 'Annotation_files/'))
    index = 0
    for filename in os.listdir(path + 'Annotation_files/'):
       print(f'{index}/{len_list}')
       with open(f'{path}Annotation_files/{filename}', 'r') as f:
           first_line = int(f.readline())
           second_line = int(f.readline())
           frame_list.append({
                'start': first_line,
                'end': second_line,
                'data': pose_point(f'{path}Videos/{filename[:-3]}avi', python.vision.RunningMode.VIDEO, webcam=False)
           })
       index = index + 1
    pd.DataFrame(frame_list).to_csv(f'{base_path}dataset_landmarks.csv', index=False)


def build_all_dataset():
    build_dataset_landmarks('./data/archive/', base_path='Coffee_room_01')
    build_dataset_landmarks('./data/archive/', base_path='Coffee_room_02')
    build_dataset_landmarks('./data/archive/', base_path='Home_01')
    build_dataset_landmarks('./data/archive/', base_path='Home_02')

build_all_dataset()