# Big Data Classification with RSP, PCA & Ensemble Learning

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange.svg)](https://scikit-learn.org/stable/)

This repository contains my **CSE 4460 Big Data Analytics Lab** project.
The task: *Build an efficient classification model for a high-dimensional big dataset, reduce dimensionality, and evaluate performance.*

We used the **Human Activity Recognition (HAR) with Smartphones** dataset (561 features, 6 activity classes) and applied:

* **Random Sample Partitioning (RSP)** → randomized block partitioning for scalability
* **Principal Component Analysis (PCA)** → dimensionality reduction
* **Classification Models** → Logistic Regression, Random Forest
* **Ensemble Learning** → soft voting across models for improved stability
* **Visualization & Reporting** → accuracy plots, confusion matrices, PCA variance

---

## Pipeline Overview

```
Raw HAR Dataset
   │
   ├── Merge train.csv + test.csv
   │
   ├── Random Sample Partitioning (RSP)
   │     └── randomized blocks
   │
   ├── PCA (fixed n_components)
   │     └── dimension reduction
   │
   ├── Train classifiers per block
   │     ├── Logistic Regression
   │     └── Random Forest
   │
   ├── Ensemble (soft voting)
   │
   └── Evaluation
         ├── Per-block metrics
         ├── Aggregate stats
         └── Global merged results
```

---

## Repository Structure

```
CSE4460_BigData_Lab1_Classification/
├── data/
│   ├── train.csv / test.csv     # Original Kaggle data
│   ├── full_merged.csv          # Combined dataset
│   ├── blocks/                  # RSP block splits
│   └── pca_blocks/              # PCA-reduced block data
├── models/                      # Trained per-block models
├── notebooks/                   # Jupyter workflow (Steps 1–6)
│   ├── 01_data_loading.ipynb
│   ├── 02_sampling_rsp_partitioning.ipynb
│   ├── 03_dimensionality_reduction.ipynb
│   ├── 04_model_training.ipynb
│   ├── 05_ensembling.ipynb
│   └── 06_visualization_and_global_results.ipynb
├── reports/
│   ├── tables/                  # Accuracy tables, reports
│   └── figures/                 # Accuracy plots, confusion matrices, PCA plots
├── README.md
└── LICENSE
```

---

## Setup Instructions

1. **Clone repository**

```bash
git clone https://github.com/<your-username>/CSE4460_BigData_Lab1_Classification.git
cd CSE4460_BigData_Lab1_Classification
```

2. **Create environment** (Anaconda recommended)

```bash
conda create -n bigdata_lab python=3.10
conda activate bigdata_lab
pip install jupyter numpy pandas matplotlib seaborn scikit-learn
```

3. **Run Notebooks** in order (`notebooks/01_...` → `06_...`).

---

## Results

* **Per-Block Accuracies:** (see `reports/tables/step5_accuracy_per_block_with_ensemble.csv`)
* **Aggregate Stats:** (mean/std/min/max across blocks)
* **Global Accuracy Summary:** (see `reports/tables/step6_global_accuracy_summary.csv`)
* **Confusion Matrices:** per-block + global ensemble (`reports/figures/...`)
* **PCA Variance Plot:** justification for dimensionality reduction

Example visualization:

> ![Per-block accuracy trends](reports/figures/step6_accuracy_trends_line.png)

---

## Report

The full project report (with methodology, figures, tables, and discussion) is included in:
-> `reports/BigDataLab_Report.pdf` (or Google Docs export)

---

## Key Takeaways

* RSP ensures scalable, randomized sampling for big data.
* PCA reduces dimensionality while preserving most variance.
* Ensemble learning consistently outperforms single classifiers.
* Final ensemble accuracy: **89.60%** (global merged evaluation).

---

*Developed as part of CSE 4460 Big Data Analytics Lab (Fall 2025).*
