from tracking.sort import Sort
from ultralytics import YOLO
import torch
import torchvision
import pathlib
from tqdm import tqdm as progress_bar
from os import system, listdir
import shutil
import sys
import os
from argparse import ArgumentParser
import numpy as np
from random import randint

def extract_frames(video_file,  out_dir = f"data/frames/"):
    '''
    Extracts frames and saves to given directory.
    Return path to result directory
    '''
    out_dir = out_dir + pathlib.Path(video_file).stem + "/"
    print(out_dir)

    # create directories for original images and results
    pathlib.Path(out_dir + "original/").mkdir(parents=True, exist_ok=True)
    pathlib.Path(out_dir + "result/").mkdir(parents=True, exist_ok=True)
    pathlib.Path(out_dir + "kitti/").mkdir(parents=True, exist_ok=True)
    
    # extract frames and save to directory
    system(f"ffmpeg -i {video_file} {out_dir}original/frame%04d.png")#C:\\ffmpeg-2024-05-06-git-96449cfeae-essentials_build\\bin\\ffmpeg.exe

    print(f"Frames saved to {out_dir}original/")

    return out_dir


def save_kitti_format(tracks, output, classes):
    '''
    Save tracking results in KITTI format
    '''
    dir = os.path.dirname(output)
    if not os.path.exists(dir):
        os.makedirs(dir)
    with open(output, 'w') as f:
        for frame_id, (tracked_bboxes, cls) in enumerate(zip(tracks, classes)):
            for bbox, cl in zip(tracked_bboxes, cls):
                x1, y1, x2, y2, obj_id, _ = bbox
                if cl not in {"car", "pedestrian"}:
                    cl = "DontCare"
                # KITTI format: frame, track_id, type, truncated, occluded, alpha, bbox_left, bbox_top, bbox_right, bbox_bottom, height, width, length, location_x, location_y, location_z, rotation_y
                line = f"{frame_id} {int(obj_id)} {cl.replace(' ', '_')} 0 0 -1 {x1} {y1} {x2} {y2} -1 -1 -1 -1000 -1000 -1000 -10\n"
                f.write(line)

def tracking(video, model_variant, filter, max_age, min_hits, iou_threshold, framerate, out):
    # Get frames
    frames_path = extract_frames(video)

    # Load a model
    if model_variant == 'pretrained':
        model = YOLO('yolov8n.pt')  # pretrained YOLOv8n model
    elif model_variant == 'finetuned':
        model = YOLO('runs/detect/train/train4/weights/best.pt')

    # get list of frames
    frames = sorted(listdir(frames_path + "original/"))

    tracker = Sort(
        max_age=max_age,
        min_hits=min_hits,
        iou_threshold=iou_threshold,
        filter_type=filter,  # kalman | ukf
        framerate=framerate
    )
    tracks = []
    classes = []

    #drawing bounding boxes parameter
    colors = {}

    for i, frame in enumerate(progress_bar(frames)):
        # Object detection
        results = model(f"{frames_path}original/{frame}", verbose=False)[0]  # return a list of Results objects
        result_boxes = torch.cat((results.boxes.xyxy.int(), results.boxes.conf.view(-1, 1)), dim=1).cpu().numpy() # concat score/confidence as last elem of tensor containing bbox

        n = result_boxes.shape[0]
        ids = np.arange(n).reshape(-1, 1)
        result_boxes = np.hstack((result_boxes, ids))

        tmp_classes = results.boxes.cls.cpu().numpy().tolist()
        tmp_classes = [results.names[int_class] for int_class in tmp_classes]
        #print(results.names)
        
        # Tracking
        tracked_bboxes = tracker.update(result_boxes)

        new_classes = [tmp_classes[int(i)] for i in tracked_bboxes[:, -1]]

        # Saving results
        bboxes = torch.from_numpy(tracked_bboxes[:, :-2])
        ids = tracked_bboxes[:, -2].astype(int).astype(str).tolist()

        labels = [f"{name} {id}" for name, id in zip(new_classes, ids)]
        for label in labels:
            if label not in colors:
                colors[label] = (randint(0, 255), randint(0, 255), randint(0, 255))
        frame_colors = [colors[label] for label in labels]
        
        image = torchvision.io.read_image(f"{frames_path}original/{frame}")
        image = torchvision.utils.draw_bounding_boxes(image, bboxes, labels=labels, width=3, font="utils/ARIAL.TTF", font_size=30, colors=frame_colors)
        image = torchvision.transforms.functional.to_pil_image(image)
        image.save(f"{frames_path}/result/{frame}")

        tracks.append(tracked_bboxes)
        classes.append(new_classes)

    shutil.rmtree(frames_path + "original/")
   
    # Save in KITTI format
    save_kitti_format(tracks, out, classes)

    # Generate recording from frames
    system(f"ffmpeg -framerate 16 -pattern_type glob -i '{frames_path}result/*.png' -c:v libx264 -pix_fmt yuv420p {frames_path}result/{model_variant}_{filter}.mp4 -y")
    for item in os.listdir(f"{frames_path}result/"):
        if item.endswith(".png"):
            os.remove(os.path.join(f"{frames_path}result/", item))

def main(ar=None):
    args = get_args(ar)
    out = f"TrackEval/data/trackers/kitti/kitti_2d_box_train/{args.filter}/{pathlib.Path(args.video).stem}.txt"
    tracking(args.video, args.model, args.filter, args.max_age, args.min_hits, args.iou_threshold, args.framerate, out)


def get_args(args=None):
    parser = ArgumentParser(description="Mutli-object tracking on given reocrding",
                            epilog="")

    parser.add_argument('video', type=str, help="Filter version, can be 'kalman' or 'ukf")
    parser.add_argument('--model', type=str, default='pretrained', help="Model variant, can be 'pretrained' or 'finetuned'")
    parser.add_argument('--filter', type=str, default='kalman', help="Filter version, can be 'kalman' or 'ukf")
    parser.add_argument('--max_age', type=int, default=4, help="Tracker max_age")
    parser.add_argument('--min_hits', type=int, default=2, help="Tracker min_hits")
    parser.add_argument('--iou_threshold', type=float, default=0.2, help="Tracker iou_threshold")
    parser.add_argument('--framerate', type=int, default=30, help="Video framerate")

    return parser.parse_args(args)

if __name__ == "__main__":
    main()
    
