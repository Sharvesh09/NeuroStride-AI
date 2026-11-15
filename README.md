# NeuroStride AI – freeze-like gait detection (research prototype)
**Not a medical device. Not clinically validated.** It flags movement patterns that resemble freezing; it does not diagnose FoG.

## What is real here
* Webcam -> MediaPipe Pose -> 6 scale-normalised gait channels -> 4 s moving window -> 42 features -> Random Forest.
* An alert needs **3 consecutive positive windows** (never one frame) with a 10 s cool-down.
* **No trained model ships with this project.** You create one from your own labelled recordings (below). Until then the dashboard tracks pose and gait metrics but classification is disabled; it never fakes predictions.
* Confidence is hidden unless a model is flagged calibrated (the trainer saves `calibrated=False`).

## Two pipelines (different sensor types, never mixed)
| | Daphnet (accelerometer) | Webcam pose |
|---|---|---|
| Data | 3 wearable accelerometers, 64 Hz, 10 patients | Your recordings via `collect_data.py` |
| Labels | 1 = no freeze, 2 = freeze, 0 = ignored | 0 = normal, 1 = freeze-like (**acted** by volunteers) |
| Validation | Leave-one-participant-out | Leave-one-recording-group-out |
| Command | `python train_model.py --mode daphnet` | `python train_model.py --mode pose` |
The Daphnet model is an accelerometer baseline and is **not** used by the webcam dashboard. A webcam model trained on acted freezing is a proof of concept, not evidence of patient performance.

## Windows setup (Python 3.10 or 3.11, 64-bit; tick "Add Python to PATH")
```
cd NeuroStride_AI
py -3.11 -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```
## Daphnet (optional baseline)
Download "Daphnet Freezing of Gait" from the UCI Machine Learning Repository (search that title), unzip, and copy the files `S01R01.txt ...` into `dataset\daphnet\`. Then `python train_model.py --mode daphnet`.
## Record webcam data (needed for the live model)
Stand 2-3 m from the camera, side or front view, whole body visible, good light.
```
python collect_data.py --subject P01 --session s1
```
Keys: **N** = normal walking, **F** = acted freeze-like (feet planted, trembling in place, stop-start), **Q** = save. Record each person several times with both labels. Use at least 3 different people (IDs P01, P02, P03...); the split is by person, so too few people means no valid evaluation.
## Train, then run
```
python train_model.py --mode pose
streamlit run app.py
```
Outputs: `models\pose_model.pkl`, `outputs\graphs\confusion_matrix_pose.png`, `outputs\reports\metrics_pose.json` (accuracy, precision, recall, F1).
## Troubleshooting
* `module 'mediapipe' has no attribute 'solutions'` -> `pip install mediapipe==0.10.14`.
* Camera not opening -> close Zoom/Teams, change camera index in the sidebar (0, 1, 2).
* `numpy` / `_ARRAY_API` errors -> `pip install "numpy<2"`.
* Status stays on "Collecting data" -> legs must be visible; step back.
* Execution policy error on activate -> `Set-ExecutionPolicy -Scope Process RemoteSigned`.
## Status of this code
Training/evaluation was smoke-tested on synthetic signals only (not reported as results). The webcam/Streamlit parts were syntax-checked but **not run on a real webcam by the author of this code**; test them on your laptop and report errors.
