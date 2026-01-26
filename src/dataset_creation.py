import os

import pandas as pd

from pose_landmark import pose_video_dataset


def build_dataset_landmarks(directory: str, base_path: str, quality="medium"):
    frame_list = []
    path = directory + base_path + '/' + base_path + '/'
    len_list = len(os.listdir(path + 'Annotation_files/'))
    index = 1
    for filename in os.listdir(path + 'Annotation_files/'):
        print(f'{index}/{len_list}')
        with open(f'{path}Annotation_files/{filename}', 'r') as f:
            try:
                first_line = int(f.readline())
                second_line = int(f.readline())
                data = pose_video_dataset(f'{path}Videos/{filename[:-3]}avi', quality)
            except Exception as _:
                print('Skipping video with multiple fall')
                data = None

            frame_list.append({
                'name': filename[:-4],
                'start': first_line,
                'end': second_line,
                'data': data
            })
        index = index + 1
    return frame_list


def build_all_dataset(quality="medium"):
    frame_list = list()
    frame_list.append(build_dataset_landmarks('../data/archive/', base_path='Coffee_room_01', quality=quality))
    frame_list.append(build_dataset_landmarks('../data/archive/', base_path='Coffee_room_02', quality=quality))
    frame_list.append(build_dataset_landmarks('../data/archive/', base_path='Home_01', quality=quality))
    frame_list.append(build_dataset_landmarks('../data/archive/', base_path='Home_02', quality=quality))

    pd.DataFrame(frame_list).to_json('../data/archive.json', index=False)


if __name__ == '__main__':
    build_all_dataset(quality="high")
