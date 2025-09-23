A forked of jahongir7174/YOLOv11-pt(https://github.com/jahongir7174/YOLOv11-pt/tree/master) for inference.

## Installation Guide

#### Prerequisites
- Python 3.10 or higher

#### Create Virtual Environment
```bash
# Create virtual environment
python -m venv ENV
```

# Activate virtual environment
##### On Windows:
```bash
ENV\Scripts\activate.bat
```
##### On macOS/Linux:
```bash
source ENV/bin/activate
```

#### Install Dependencies
```bash
pip install -r requirements.txt
```

#### Run the Application
```bash
main_inference.ipynb
```

### Inference:
* Added new image for inference to [dataset]
* Change the spesific image you want to inference in main_inference.py ("IMG_PATH" variable). Can ONLY inference one image at a time.

