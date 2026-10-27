# Machine Learning Assignment 2 - NASA Near-Earth Objects (NEO) Hazard Classifier

## a. Problem Statement
The goal of this project is to classify Near-Earth Objects (NEOs) as **Potentially Hazardous Asteroids (PHA)** (`is_potentially_hazardous_asteroid`) using physical dimensions, brightness, orbital velocity, and Keplerian orbital parameters.

## b. Dataset Description
- **Dataset Name**: NASA Near-Earth Objects (NEO) Dataset
- **Source**: [Kaggle Dataset Link](https://www.kaggle.com/datasets/sunethjayawardana/nasa-near-earth-objects-neo-full-dataset?resource=download)
- **Dataset Variable**: `neo_dur_df`
- **Instances**: 25,869 objects
- **Features**: 44 features
- **Target Variable**: `is_potentially_hazardous_asteroid` (0 = Non-Hazardous, 1 = Potentially Hazardous)

## c. GitHub Repository Link
[https://github.com/2026aj05209-ayan/MLAssigment2](https://github.com/2026aj05209-ayan/MLAssigment2)

## d. Model Evaluation Metrics Comparison Table
| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Logistic Regression** | 0.9679 | 0.9900 | 0.8450 | 0.8180 | 0.8313 | 0.8137 |
| **Decision Tree** | **0.9973** | **0.9956** | **0.9880** | **0.9840** | **0.9860** | **0.9845** |
| **KNN** | 0.9322 | 0.9334 | 0.6987 | 0.5240 | 0.5989 | 0.5696 |
| **Naive Bayes** | 0.9509 | 0.9781 | 0.7189 | 0.8080 | 0.7608 | 0.7351 |
| **Support Vector Machine** | 0.9710 | 0.9922 | 0.8755 | 0.8160 | 0.8447 | 0.8294 |

### Observations on Model Performance
| ML Model Name | Observation about model performance |
|---|---|
| **Logistic Regression** | Strong linear baseline (**96.79% Accuracy**, **0.9900 AUC**) after feature scaling. |
| **Decision Tree** | Achieves top performance across all 6 metrics (**99.73% Accuracy**, **0.9956 AUC**, **0.9860 F1**, **0.9845 MCC**). |
| **KNN** | Delivers lower overall performance (**93.22% Accuracy**, **0.5240 Recall**) due to distance distortion across high-dimensional features. |
| **Naive Bayes** | Solid recall (**80.80%**) and AUC (**0.9781**), effective at detecting hazards despite conditional independence assumptions. |
| **Support Vector Machine** | Second-best performing model (**97.10% Accuracy**, **0.9922 AUC**, **0.8447 F1**) using an RBF kernel. |
| **Overall Winner** | **Decision Tree Classifier** |
