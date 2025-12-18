import os
import pandas as pd
from pose_landmark import pose_point
from mediapipe.tasks import python

def build_dataset_landmarks(directory: str, base_path:str):
    frame_list = []
    path = directory + base_path + '/' + base_path + '/'
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
    return frame_list


def build_all_dataset():
    frame_list = list()
    frame_list.append(build_dataset_landmarks('./data/archive/', base_path='Coffee_room_01'))
    frame_list.append(build_dataset_landmarks('./data/archive/', base_path='Coffee_room_02'))
    frame_list.append(build_dataset_landmarks('./data/archive/', base_path='Home_01'))
    frame_list.append(build_dataset_landmarks('./data/archive/', base_path='Home_02'))

    pd.DataFrame(frame_list).to_csv('./data/archive.csv', index=False)

if __name__ == '__main__':
    build_all_dataset()