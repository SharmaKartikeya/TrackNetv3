import os
import csv

yolo_annotations = 'datasets/video_dataset/yolo_annotations'
csvs = 'datasets/video_dataset/csvs'
videos = 'datasets/video_dataset/videos'

fields = ['frame_num', 'x', 'y', 'w', 'h', 'visible']

for video in os.listdir(videos):
    name = video.split('.')[0]
    yolo_ann_dir = os.path.join(yolo_annotations, name)
    obj_train_data_path = os.path.join(yolo_ann_dir, 'obj_train_data')

    csv_path = os.path.join(csvs, name + '.csv')
    csv_file = open(csv_path, "w")
    csvwriter = csv.writer(csv_file)
    csvwriter.writerow(fields)
    
    for idx, txt_file in enumerate(sorted(os.listdir(obj_train_data_path))):
        txt_file_path = os.path.join(obj_train_data_path, txt_file)
        row = list()
        if(os.stat(txt_file_path).st_size == 0):
            row = [idx, 0, 0, 0, 0, 0]

        else:        
            with open(txt_file_path) as f:
                line = f.readline()   
                x, y, w, h = line.split(' ')[1:]
                h = h.strip()
                row = [idx, x, y, w, h, 1]

        csvwriter.writerow(row)

    print(f"Created csv file for {video}")
