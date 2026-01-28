# CAL303 – Deep Learning Lab

This repository contains implementations of multiple Deep Learning lab experiments
using a common preprocessing pipeline and reusable training utilities.

The project is structured to keep code lightweight and reproducible, while handling
datasets locally (not uploaded to GitHub).

---

## 📁 Project Structure

```

CAL303_Deep_Learning_Lab/
├── data/
│   ├── raw/            # Raw datasets (NOT tracked by git)
│   ├── processed/      # Preprocessed data (generated locally)
│   └── preprocess.py   # Generic preprocessing pipeline
│
├── experiments/
│   ├── exp01.ipynb
│   ├── exp02.ipynb
│   ├── exp03.ipynb
│   └── exp04.ipynb
│
├── utils/
│   ├── models.py
│   ├── train.py
│   └── metrics.py
│
├── requirements.txt
└── README.md

```

---

## 📦 Dataset Handling (Important)

Datasets are **NOT uploaded to GitHub** to keep the repository lightweight.

### How to add a dataset

1. Download the dataset provided by the instructor / Kaggle / UCI.
2. Place the dataset file inside:

```

data/raw/dataset.csv

````

**Assumption:**  
- Dataset format: CSV  
- Last column = target variable  

---

## 🔄 Data Preprocessing

A generic preprocessing pipeline is provided.

It automatically supports:
- classification
- regression
- numerical + categorical features

### Run preprocessing

```bash
python data/preprocess.py
````

This generates:

```
data/processed/
├── X_train.npy
├── X_val.npy
├── X_test.npy
├── y_train.npy
├── y_val.npy
└── y_test.npy
```

---

## 🧪 Running Experiments

Each lab experiment is implemented as a separate Jupyter notebook.

Example:

```bash
experiments/exp01.ipynb
```

Inside the notebook, load processed data:

```python
import numpy as np

X_train = np.load("../data/processed/X_train.npy")
y_train = np.load("../data/processed/y_train.npy")
```

---

## ⚙ Environment Setup

Create and activate a virtual environment, then install dependencies:

```bash
pip install -r requirements.txt
```

---

## 📝 Notes

* Raw and processed datasets are excluded using `.gitignore`
* Code is reusable across multiple experiments
* Each notebook corresponds to one lab experiment

---

## 📚 Course

**CAL303 – Deep Learning Lab**
B.Tech CSE (AI & ML)

````

---

## ✅ Also Make Sure `.gitignore` Has This (Very Important)

```gitignore
# Data
data/raw/
data/processed/

# Python
__pycache__/
.ipynb_checkpoints/
````

