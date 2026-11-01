# lupus-nephritis-ml-xai
# Explainable Machine Learning for Predicting Lupus Nephritis Histopathological Classes

[![Research](https://img.shields.io/badge/Research-Medical%20AI-blue)]()
[![Machine Learning](https://img.shields.io/badge/Method-Machine%20Learning-orange)]()
[![Explainable AI](https://img.shields.io/badge/Explainability-SHAP-green)]()
[![Status](https://img.shields.io/badge/Status-Research-lightgrey)]()

## Overview

This repository contains the research implementation and supplementary materials for the study:

**Explainable Machine Learning Models for Predicting Lupus Nephritis Histopathological Classes Using Clinical and Laboratory Data**

Lupus nephritis (LN) is a serious complication of systemic lupus erythematosus (SLE). Histopathological classification based on kidney biopsy is essential for disease characterization and treatment planning. This study investigates whether machine-learning models can identify patterns in routinely collected clinical and laboratory variables that help predict LN histopathological classes.

We develop and evaluate machine-learning models for predicting the histopathological classes of lupus nephritis and use Explainable Artificial Intelligence (XAI) techniques to explore the contribution of individual clinical and laboratory features to model predictions.

**Important:** The models are intended for research purposes and are not designed to replace kidney biopsy, histopathological assessment, or clinical judgment.

## Study Objectives

The main objectives are:

1. Develop machine-learning models using clinical and laboratory data to predict lupus nephritis histopathological classes.
2. Evaluate multiclass classification performance for ISN/RPS Classes III, IV, and V.
3. Investigate a secondary binary classification task distinguishing Class IV from Classes III and V combined.
4. Compare the performance of different machine-learning algorithms.
5. Apply SHAP (SHapley Additive exPlanations) to interpret model predictions and identify influential features.
6. Investigate model robustness using cross-validation, bootstrap confidence intervals, and permutation testing.

## Dataset

The study uses retrospective clinical data from patients with lupus nephritis evaluated at Mashhad University of Medical Sciences, Iran.

| Characteristic | Description |
|---|---|
| Study design | Retrospective observational study |
| Total patients | 52 |
| Histopathological classes | III, IV, and V |
| Class III | 5 patients |
| Class IV | 43 patients |
| Class V | 4 patients |
| Predictors | 14 clinical and laboratory variables |

### Input Features

The models use the following clinical and laboratory variables:

- Age at first visit
- Sex
- Discoid lupus erythematosus (DLE)
- Malar rash
- Hair loss
- Arthritis
- Photosensitivity
- Oral ulcer
- Lower-limb edema
- 24-hour urinary creatinine threshold indicator
- Red blood cell (RBC) casts
- Creatinine measurement indicator
- Sex-adjusted elevated serum creatinine indicator
- Log-transformed 24-hour urinary protein

**Data privacy:** Patient-level clinical data are not included in this public repository because of patient confidentiality and ethical considerations. Any future data-sharing must comply with institutional ethical approvals and applicable data-protection requirements.

## Classification Tasks

### 1. Multiclass Classification

Prediction of three lupus nephritis histopathological classes:

- Class III
- Class IV
- Class V

### 2. Binary Classification

An exploratory binary classification task:

- Positive class: Class IV
- Negative class: Classes III and V combined

This binary task is intended to evaluate discrimination of Class IV from the other two observed classes. It should not be interpreted as a direct comparison of proliferative versus non-proliferative lupus nephritis.

## Machine Learning Methods

The study evaluates eight machine-learning configurations using clinical and laboratory predictors.

The best-performing configuration in the reported experiments was XGBoost.

Model evaluation includes:

- Cross-validation
- Receiver Operating Characteristic (ROC) analysis
- Area Under the ROC Curve (AUC)
- Balanced accuracy
- Macro-averaged F1-score
- Bootstrap confidence intervals
- Permutation testing

Performance is assessed separately for the multiclass and binary classification tasks.

## Explainable Artificial Intelligence

### SHAP Analysis

SHAP (SHapley Additive exPlanations) is used to investigate the contribution of individual features to model predictions.

Mean absolute SHAP values are used to estimate the relative magnitude of feature contributions.

The analysis is performed separately for:

- Class III predictions
- Class IV predictions
- Class V predictions
- The positive class in the binary classification task

### Main SHAP Findings

The reported SHAP analysis identified the following influential features:

| Prediction target | Most influential features |
|---|---|
| Class III | Age, urinary protein, DLE |
| Class IV | Urinary protein, age, photosensitivity |
| Class V | Urinary protein, age, lower-limb edema |
| Binary positive class | Urinary protein, age, photosensitivity |

These results suggest that age and urinary protein measurements contribute substantially to the model outputs, although their relative importance varies across prediction targets.

**Interpretation note:** Mean absolute SHAP values quantify contribution magnitude, not the direction of an association or a causal relationship. Feature importance should therefore not be interpreted as evidence that a clinical variable independently predicts a histopathological class.

## Repository Structure

The repository is organized as follows (to be updated to match the final uploaded files):

```text
lupus-nephritis-ml-xai/
│
├── README.md
│
├── data/
│   └── README.md
│
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_exploratory_data_analysis.ipynb
│   ├── 03_multiclass_classification.ipynb
│   ├── 04_binary_classification.ipynb
│   └── 05_shap_explainability.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── model_training.py
│   ├── model_evaluation.py
│   └── explainability.py
│
├── results/
│   ├── figures/
│   └── tables/
│
├── requirements.txt
│
└── LICENSE
```

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR-USERNAME/lupus-nephritis-ml-xai.git
cd lupus-nephritis-ml-xai
```

Create a Python environment:

```bash
python -m venv venv
```

Activate the environment:

**Windows**
```bash
venv\Scripts\activate
```

**Linux / macOS**
```bash
source venv/bin/activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

The exact package versions and dependencies should be documented in `requirements.txt` to support reproducibility.

## Reproducibility and Validation

The repository is intended to provide transparent documentation of the machine-learning workflow, including:

- Data preprocessing and feature definitions
- Model configuration and training procedures
- Cross-validation strategy
- Performance evaluation
- Bootstrap uncertainty estimation
- Permutation testing
- SHAP-based model interpretation

Because the dataset is small and the histopathological classes are imbalanced, model performance and feature-importance estimates should be interpreted cautiously. Independent external validation using a larger multicentre cohort is required before considering clinical implementation.

## Limitations

The study has several important limitations:

1. The sample size is relatively small, particularly for Classes III and V.
2. The distribution of histopathological classes is substantially imbalanced.
3. The retrospective single-centre design may limit generalizability.
4. SHAP explanations are model-dependent and do not establish causality.
5. The reported performance requires external validation before clinical application.

## Ethical Considerations and Data Availability

This study was conducted using retrospective clinical data. Ethical approval and data-sharing conditions should be reported in accordance with the final published manuscript.

Individual-level patient data are not publicly available because of privacy and confidentiality restrictions. Requests for access, where permitted, should be directed to the corresponding author and are subject to institutional approval.

## Citation

If you use this repository or its methods in your research, please cite the associated publication:

Explainable Machine Learning Models for Predicting Lupus Nephritis Histopathological Classes Using Clinical and Laboratory Data. 

## Contact

**Corresponding author:** Toktam Dehghani  
Assistant Professor of Medical Informatics  
Mashhad University of Medical Sciences, Iran

## License

The code license will be specified in the repository. Patient-level clinical data are not covered by the code license and are not distributed through this repository.
