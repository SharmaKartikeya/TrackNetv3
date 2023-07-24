from argparse import ArgumentParser
from graphs.models.tracknetv3 import TrackNetv3
from graphs.models.tracknetv2 import TrackNetv2
import cv2 as cv
import numpy as np
import torchvision
import torch
from utils.ball_position import get_ball_position
from utils.config import process_configs

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/predict.yaml', help='Path to prediction configuration file')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    config = process_configs([args.config])

    config.dropout = 0
    device = torch.device(config.device)
    model = TrackNetv3(config).to(device)
    model.load(config.weights, device=config.device)
    model.eval()

    cap = cv.VideoCapture(config.video)
    videoEndReached = False

    img_array = []

    counter = 0
    while cap.isOpened():
        # TODO: add one_output_frame support

        frames = []
        for _ in range(config.sequence_length):
            ret, frame = cap.read()
            counter += 1
            if not ret:
                videoEndReached = True
                break
            frames.append(frame)

        if videoEndReached:
            break

        frames_torch = []
        for frame in frames:
            frame_torch = torch.tensor(frame).permute(2, 0, 1).float().to(device) / 255
            frame_torch = torchvision.transforms.functional.resize(frame_torch, config.image_size)
            frames_torch.append(frame_torch)

        frames_torch = torch.cat(frames_torch, dim=0).unsqueeze(0)
        pred = torch.sigmoid(model(frames_torch))
        pred = pred[0, :, :, :].detach().cpu().numpy()

        for i in range(config.sequence_length):
            pred_frame = pred[i, :, :]
            pred_frame = cv.resize(pred_frame, (frames[i].shape[1], frames[i].shape[0]), interpolation=cv.INTER_AREA)
            get_ball_position(pred_frame, config, original_img_=frames[i])
            size = pred_frame.shape
            save_frame = np.uint8(pred_frame * 255)

            img_array.append(save_frame)

            if config.visualize:
                cv.imshow('prediction', pred_frame)
                cv.imshow('original', frames[i])
                cv.waitKey(config.waitBetweenFrames)

            if cv.waitKey(10) & 0xFF == ord('q'):
                break

    if config.save_path:
        out = cv.VideoWriter(config.save_path, cv.VideoWriter_fourcc(*'mp4v'), 25, (size[1], size[0]), False)
        for i in range(len(img_array)):
            out.write(img_array[i])
            print(img_array[i].min(), img_array[i].max())
        out.release()

    cap.release()
    cv.destroyAllWindows()  # destroy all opened windows
