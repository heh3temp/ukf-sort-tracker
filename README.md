# TWM



## Tracking
1. Clone `https://github.com/abewley/sort` into tracking folder: `clone https://github.com/abewley/sort.git tracking`
2. Install requirements: `pip install -r requirements.txt`
3. Run benchmark.py :

Example usage:
```bash
python benchamrk.py  recording.mp4
python benchamrk.py  recording.mp4 --model finetuned --filiter ukf --max_age 20 --iou_threshold 0.3


Available options:
video (str) - Path to recording
model (str) default = 'pretrained' - Model variant, can be 'pretrained' or 'finetuned'
filter (str) default = 'kalman' - Filter version, can be 'kalman' or 'ukf'
max_age (int) default = 4 - Tracker max_age
min_hits (int) default = 2 - Tracker min_hits
iou_threshold (float) default = 0.2 - Tracker iou_threshold
framerate (int) default = 30 - Video framerate
```


## Evaluation
1. Run tracking with max 1 argument
2. Install Python 3.8
3. Install requirements: `pip install -r TrackEval/requirements.txt` or `pip install -r TrackEval/minimum_requirements.txt`
4. Run `TrackEval/scripts/run_kitti.py`
5. Results are in folder(s) `TrackEval\data\trackers\kitti\kitti_2d_box_train\{tracker_name}`
