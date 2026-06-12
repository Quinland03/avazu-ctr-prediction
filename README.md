# Avazu Click-Through Rate Prediction Using PySpark

## Overview

This project applies machine learning techniques to predict whether a user will click on an online advertisement using the Avazu Click-Through Rate Prediction dataset from Kaggle.

The project was completed for a Big Data Analytics course and demonstrates the use of Apache Spark for large-scale data processing and machine learning. Multiple supervised learning models were trained and evaluated on a dataset containing over 40 million advertising impressions.

---

## Dataset

### Source

Avazu Click-Through Rate Prediction Competition

https://www.kaggle.com/competitions/avazu-ctr-prediction

### Dataset Description

The dataset contains advertising impression records collected from mobile advertisements. The goal is to predict whether a displayed advertisement will be clicked.

### Dataset Statistics

| Metric             | Value      |
| ------------------ | ---------- |
| Rows               | 40,428,967 |
| Features           | 24         |
| Dataset Size       | 1.28 GB    |
| Clicks             | 6,865,066  |
| Non-Clicks         | 33,563,901 |
| Click-Through Rate | 16.98%     |

### Target Variable

* click = 1 → Advertisement clicked
* click = 0 → Advertisement not clicked

### Selected Features

* banner_pos
* site_category
* app_category
* device_type
* device_conn_type
* hour_of_day
* C1
* C14
* C15
* C16
* C17
* C18
* C19
* C20
* C21

---

## Technologies

* Python
* Apache Spark (PySpark)
* Pandas
* Matplotlib
* Scikit-Learn
* Jupyter Notebook

---

## Exploratory Data Analysis

### Overall Click-Through Rate

CTR = 16.98%

Only a small fraction of advertisements receive clicks, indicating a moderately imbalanced classification problem.

### CTR by Hour

User engagement varied throughout the day.

Highest observed CTR:

* 01:00 → 18.58%
* 15:00 → 18.12%
* 00:00 → 18.04%

Lowest observed CTR:

* 09:00 → 16.01%
* 20:00 → 16.05%
* 21:00 → 16.07%

### CTR by Banner Position

Banner position had a measurable impact on click probability.

| Banner Position | CTR    |
| --------------- | ------ |
| 0               | 16.43% |
| 1               | 18.36% |
| 7               | 32.01% |

### CTR by Device Type

Different device types exhibited different click behaviors.

| Device Type | CTR    |
| ----------- | ------ |
| 0           | 21.07% |
| 1           | 16.92% |
| 4           | 9.54%  |
| 5           | 9.38%  |

---

## Methodology

Apache Spark was used to process the large-scale dataset and train machine learning models.

A 5% random sample of the dataset was used for model development and evaluation, resulting in approximately 2 million observations.

### Data Preparation

1. Load data using Spark DataFrames
2. Extract hour of day from timestamp
3. Encode categorical variables using:

   * StringIndexer
   * OneHotEncoder
4. Assemble features using VectorAssembler
5. Split data into training and testing sets

### Evaluation Metrics

Models were evaluated using:

* Area Under the ROC Curve (AUC)
* Logarithmic Loss (competition metric)

---

## Machine Learning Models

### Logistic Regression

A baseline linear classification model.

AUC: **0.6459**

### Decision Tree

A single-tree classification model.

AUC: **0.4552**

### Random Forest

An ensemble model using multiple decision trees.

AUC: **0.6495**

### Gradient Boosted Trees

An ensemble boosting algorithm that sequentially improves weak learners.

AUC: **0.6895**

---

## Model Performance

| Model                  | AUC    |
| ---------------------- | ------ |
| Decision Tree          | 0.4552 |
| Logistic Regression    | 0.6459 |
| Random Forest          | 0.6495 |
| Gradient Boosted Trees | 0.6895 |

### Best Model

Gradient Boosted Trees achieved the highest predictive performance with an AUC of 0.6895.

---

## Repository Structure

```text
avazu-ctr-prediction/
│
├── README.md
├── requirements.txt
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── EDA.ipynb
│   └── Model_Development.ipynb
│
├── src/
│   ├── load_data.py
│   ├── eda.py
│   ├── train_logistic.py
│   ├── train_tree.py
│   ├── train_rf.py
│   ├── train_gbt.py
│   ├── evaluate.py
│   └── plot_results.py
│
├── results/
│   ├── metrics.csv
│   └── model_comparison_auc.png
│
└── report/
```

---

## Key Findings

* The dataset contains over 40 million observations and is appropriate for big data analytics.
* Advertisement click behavior varies by time of day, device type, and banner position.
* Decision Trees performed poorly on this high-dimensional classification task.
* Ensemble methods significantly improved prediction performance.
* Gradient Boosted Trees produced the strongest results among the tested models.

---

## Future Work

Potential improvements include:

* Hyperparameter tuning
* Additional feature engineering
* Training on larger samples
* Full dataset model training
* Deep learning approaches for CTR prediction
* Real-time advertising prediction systems

---

## Author

Quinlan Wilson

University of California, Santa Barbara

PSTAT Big Data Analytics Final Project
