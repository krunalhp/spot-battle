# Spot Battle: Puzzle Image Generator

Offline puzzle generator for "find the difference" images and videos.

The project includes:

- A Streamlit app for automatic and guided puzzle generation
- A PySide6 desktop editor for object selection, manual edits, and undo
- Full HD MP4 video export with a timed progress bar and solution reveal loop
- A local SAM-based segmentation pipeline that runs without external APIs

## Features

- Upload a cartoon or illustrated image and detect candidate objects with Meta SAM
- Generate puzzle and solution images automatically
- Manually select detected objects and apply subtle changes
- Edit puzzles in a desktop UI with object outlines and undo support
- Export `puzzle.png`, `solution.png`, and `puzzle_video.mp4`
- Configure puzzle video duration, default `90` seconds

## Project Structure

```text
app.py
editor_app.py
ftd/
tests/
.streamlit/
models/
outputs/
requirements.txt
```

## Prerequisites

- Python 3.10 or newer
- `git`
- Enough disk space for PyTorch, PySide6, and the SAM checkpoint

## Setup On Any Machine

### 1. Clone the repository

```bash
git clone -b puzzle-image-generator https://github.com/krunalhp/spot-battle.git
cd spot-battle
```

### 2. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
```

Linux or macOS:

```bash
python3 -m venv .venv
```

### 3. Install dependencies

Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Linux or macOS:

```bash
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r requirements.txt
```

### 4. Download the SAM checkpoint

Create the `models` directory if it does not exist, then download:

Windows PowerShell:

```powershell
New-Item -ItemType Directory -Force models
curl.exe -L -o models\sam_vit_b_01ec64.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

Linux or macOS:

```bash
mkdir -p models
curl -L -o models/sam_vit_b_01ec64.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

## Run The Streamlit App

Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Linux or macOS:

```bash
.venv/bin/python -m streamlit run app.py
```

## Run The Desktop Editor

Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe editor_app.py
```

Linux or macOS:

```bash
.venv/bin/python editor_app.py
```

The desktop editor supports:

- Loading an image
- Detecting objects and drawing selection boundaries
- Clicking objects directly on the canvas
- Applying manual edits like recolor, remove, shift, resize, and flip
- Undoing changes
- Saving edited images, puzzle images, solution images, and videos

## Run Tests

Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Linux or macOS:

```bash
.venv/bin/python -m pytest
```

## Linux Notes

- PySide6 may require desktop GUI libraries depending on the distro
- On headless Linux servers, use the Streamlit workflow unless a GUI session is available
- CPU-only PyTorch is recommended unless you intentionally want a CUDA build

## Outputs

- `puzzle.png`: side-by-side puzzle image
- `solution.png`: side-by-side solution image with highlighted differences
- `puzzle_video.mp4`: 1920x1080 countdown video with progress bar and final reveal loop

## Notes

- The repository does not include the SAM checkpoint file
- Generated outputs and local virtual environments are intentionally git-ignored
- The app works best with clean cartoon or illustrated images that contain clearly separated objects
