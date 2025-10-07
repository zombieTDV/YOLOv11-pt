A forked of jahongir7174/YOLOv11-pt(https://github.com/jahongir7174/YOLOv11-pt/tree/master) for inference and evalution on 2017 COCO offical validation data set.

## Installation Guide

#### Prerequisites
- Python 3.10 or higher

#### Create Virtual Environment
```bash
# Create virtual environment
python -m venv ENV
```

#### Activate virtual environment

##### On Windows:

```bash
venv\Scripts\activate.bat
```

##### On macOS/Linux:

```bash
source venv/bin/activate
```

#### Install Dependencies

```bash
pip install -r requirements.txt
```

#### Inference result can be view in

```bash
main_inference.ipynb
```

### How to Inference?

1. Added your own images for inference to "./dataset"
2. specify images that you want to do inference in main_inference.py ("IMG_PATH" variable). Can inference a list of images for each run.

