A forked of jahongir7174/YOLOv11-pt(https://github.com/jahongir7174/YOLOv11-pt/tree/master) for inference and evalution on 2017 COCO offical validation data set.

## Installation Guide

### Prerequisites
- Python **3.9 or higher** (3.12 recommended)
- Git installed

---

### Step 1: Clone the Repository

```bash
git clone https://github.com/zombieTDV/YOLOv11-pt.git
cd YOLOv11-pt
```

### Step 2: Create Virtual Environment

```bash
python -m venv .venv
```

Activate the virtual environment:

- On Windows (PowerShell):

```bash
  .venv\Scripts\Activate.ps1
```

- On macOS/Linux:

```bash
  source .venv/bin/activate
```

---

### Step 3: Install the Project

Development mode (recommended):

```bash
pip install -e .[dev]
pip install -r requirements.txt
```

Runtime only (minimal install):

```bash
pip install -e .
```

---

#### Evaluation result can be view in

```bash
evaluation.ipynb
```

### How to do Evalution yourself?

1. Download the COCO val dataset (or any other variants), like this [2017 val](http://images.cocodataset.org/zips/val2017.zip)
2. Download the correspond labels, for example, if you downloaded the 2017 val, then it labels are [2017 val annotations](http://images.cocodataset.org/annotations/stuff_annotations_trainval2017.zip)
3. Make sure that you put those downloaded folder in the correct name that specifile under **CONFIG** section of *evaluation.ipynb*