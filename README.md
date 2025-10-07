A forked of jahongir7174/YOLOv11-pt(https://github.com/jahongir7174/YOLOv11-pt/tree/master) for evalution (make sure to switch branch to **clean-evaluation**) on 2017 COCO offical validation data set.

## Installation Guide

#### Prerequisites
- Python 3.10 or higher

#### Clone the repository

```bash
git clone https://github.com/zombieTDV/YOLOv11-pt.git
cd YOLOv11-pt
```

#### Create Virtual Environment
```bash
# Create virtual environment
python -m venv venv
```

#### Activate virtual environment

##### On Windows:

```bash
venv\Scripts\activate.bat
```

##### On macOS/Linux:

```bash
source venv/Scripts/Activate.ps1
```

#### Install Dependencies

```bash
pip install -r requirements.txt
```

#### Evaluation result can be view in

```bash
evaluation.ipynb
```

### How to do Evalution yourself?

1. Download the COCO val dataset (or any other variants), like this [2017 val](http://images.cocodataset.org/zips/val2017.zip)
2. Download the correspond labels, for example, if you downloaded the 2017 val, then it labels are [2017 val annotations](http://images.cocodataset.org/annotations/stuff_annotations_trainval2017.zip)
3. Make sure that you put those downloaded folder in the correct name that specifile under **CONFIG** section of *evaluation.ipynb*