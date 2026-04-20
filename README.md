# Bag Guard System (BGS)

Final project implementation for unattended bag monitoring using object detection, tracking, re-identification, and ownership logic.

## Overview

This repository contains the maintained Bag Guard System codebase. The system detects people and supported bag classes, tracks them across frames, re-identifies them when tracking becomes unstable, estimates owner proximity, and raises unattended-bag states over time.

## Core Capabilities

- YOLO-based person and bag detection
- DeepSORT-based multi-object tracking with ByteTrack still available as a fallback backend
- person re-identification with OSNet and `torchreid`
- bag re-identification with learned appearance embeddings plus persistent cosine-similarity matching
- ownership assignment using distance and temporal persistence rules
- unattended-bag state classification with `OK`, `POTENTIAL`, and `UNATTENDED`
- persistent logging of person and bag identities across runs
- modular architecture with separate components for CLI, system flow, re-ID, ownership, distance estimation, and visualization

## Project Structure

- `src/main.py` - thin entrypoint
- `src/bgs/cli.py` - command-line parsing and path resolution
- `src/bgs/system.py` - main runtime orchestration and frame pipeline
- `src/bgs/reid.py` - person and bag re-identification plus persistence
- `src/bgs/ownership.py` - owner assignment and unattended-state logic
- `src/bgs/distance.py` - monocular distance and 3D position estimation
- `src/bgs/visualization.py` - drawing and debug overlay logic
- `models/` - canonical model weight directory
- `models/reid/` - re-identification weights
- `data/` - sample test videos
- `trackers/` - tracker configuration presets
- `outputs/` - generated output videos
- `logs/` - persistent re-identification data and frame logs

## Requirements

- Python 3.11
- dependencies installed from the workspace root with `pip install -r requirements.txt`

## Runtime Assets

- detector model: `models/yolo26x.pt`
- person re-ID model: `models/reid/osnet_x1_0_msmt17.pt`
- ByteTrack main preset: `trackers/bytetrack_bgs.yaml`
- ByteTrack stable preset: `trackers/bytetrack_bgs_stable.yaml`

## Running

From this project directory:

```bat
python .\src\main.py --source 0 --show
```

Run on a sample video:

```bat
python .\src\main.py --source .\data\uni.mp4 --tracker-profile stable
```

Run with ByteTrack explicitly:

```bat
python .\src\main.py --source .\data\uni.mp4 --tracker-backend bytetrack --tracker-profile stable
```

Show CLI options:

```bat
python .\src\main.py --help
```

## CLI Options

- `--source` webcam index or video path
- `--model` detector model path, defaulting to `models/yolo26x.pt`
- `--out` output video path
- `--show` display live preview
- `--imgsz` inference image size
- `--max_fps` processing FPS cap
- `--skip` process every `skip + 1` frame
- `--half` enable FP16 when CUDA is available
- `--detector-runtime` choose `torch` or `openvino`
- `--openvino-device` choose `auto`, `cpu`, or `gpu` when using OpenVINO
- `--tracker-backend` choose `deepsort` or `bytetrack`
- `--tracker-profile` choose `main` or `stable` when using ByteTrack

OpenVINO example on Intel hardware:

```bat
python .\src\main.py --source 0 --show --detector-runtime openvino --openvino-device auto
```

## Outputs

- processed video output defaults to `outputs/detection_output.mp4`
- person persistence is stored in `logs/person_reid/`
- bag persistence is stored in `logs/bag_reid/`

## Design Notes

- the maintained project no longer depends on the archived `V5` prototype folder
- final detector and re-ID weights are centralized under `models/`
- `src/main.py` is intentionally small and delegates to the `bgs` package
- the codebase has been refactored to reduce duplication and keep major responsibilities separated

## Known Limitations

- low effective FPS can increase ID switches because fewer reliable updates reach the tracker and re-ID logic
- identity recovery still depends on crop quality, so blur, occlusion, and missed detections reduce reliability
- ownership association is proximity-based and can be confused by dense crowds or nearby people
- monocular distance estimation is approximate and depends on assumed object dimensions rather than camera calibration
- several thresholds are tuned for this project setup and may need adjustment for different cameras or scenes

## Status

This folder is the canonical final implementation. Archived prototypes and legacy assets have been moved to the workspace-level `experiments/` directory.
