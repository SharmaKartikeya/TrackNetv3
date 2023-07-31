import os

folder_paths = ['datasets/video_dataset/yolo_annotations/Cam_0']

for folder in folder_paths:
    # Each folder must have an `obj_train_data` sub directory in it
    train_data_dir = os.path.join(folder, 'obj_train_data')
    for filename in os.listdir(train_data_dir):
        if filename.endswith(".txt"):
            prefix = folder.split('/')[-1]
            file_path = os.path.join(train_data_dir, filename)
            new_filename = filename.replace("frame", prefix)
            new_file_path = os.path.join(train_data_dir, new_filename)
            os.rename(file_path, new_file_path)

    print(f"renamed {folder}")