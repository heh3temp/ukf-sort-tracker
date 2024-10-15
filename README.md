# Tracking on KITTI dataset using YOLOv8 and modified SORT tracker with UKF filter



## Main Setup
1. Install python3.8 and select (e.g. by using `pyenv`)
2. Create and activate virtual environment of your choice (e.g. `python3 -m venv venv`)
3. Install numpy `pip install numpy==1.21.0`
4. Install other requirements: `pip install -r requirements.txt`
4. Run benchmark.py :



## Evaluation
1. Clone TrackEval: `git clone https://github.com/JonathonLuiten/TrackEval.git`
2. Install requirements: `pip install -r TrackEval/requirements.txt` or `pip install -r TrackEval/minimum_requirements.txt`
3. Run tracking on the entire kitti dataset
4. Run `TrackEval/scripts/run_kitti.py`
5. Results are in folder(s) `TrackEval/data/trackers/kitti/kitti_2d_box_train/{tracker_name}`

# Usage:
```bash
python benchmark.py  recording.mp4
python benchmark.py  recording.mp4 --model finetuned --filter ukf --max_age 20 --iou_threshold 0.3


Available options:
video (str) - Path to recording
model (str) default = 'pretrained' - Model variant, can be 'pretrained' or 'finetuned'
filter (str) default = 'kalman' - Filter version, can be 'kalman' or 'ukf'
max_age (int) default = 4 - Tracker max_age
min_hits (int) default = 2 - Tracker min_hits
iou_threshold (float) default = 0.2 - Tracker iou_threshold
framerate (int) default = 30 - Video framerate
```

