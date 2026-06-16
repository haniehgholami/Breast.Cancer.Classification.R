# Breast Cancer Classification Analysis

## Overview

This project presents a comparative analysis of supervised machine learning models for breast cancer diagnosis using R. The primary objective is to evaluate and compare the predictive performance of multiple classification algorithms on a diagnostic dataset.

## Dataset

The dataset contains diagnostic measurements used to classify breast cancer cases into two categories:

* Benign
* Malignant

The data file is available in this repository as `breast_cancer.csv`.

## Implemented Models

The following classification algorithms were implemented and evaluated:

1. **K-Nearest Neighbors (KNN):** Hyperparameter tuning identified **K = 7** as the optimal value.
2. **Logistic Regression**
3. **Linear Discriminant Analysis (LDA)**
4. **Quadratic Discriminant Analysis (QDA)**
5. **Naive Bayes**

## Methodology

The project workflow included:

* Data preprocessing and factor encoding
* Splitting the dataset into training (70%) and test (30%) sets
* Model training and prediction
* Hyperparameter tuning for KNN using a loop
* Performance evaluation using a custom metrics function (Accuracy, Sensitivity, Specificity)

## Results

The comparative analysis showed that **Linear Discriminant Analysis (LDA)** and **Logistic Regression** achieved the best overall performance on the test dataset, demonstrating superior classification accuracy and balanced sensitivity compared to the other models.

## Technologies Used

* R (`tidyverse`, `tidymodels`, `MASS`, `FNN`, `pROC`, `naivebayes`)
* Machine Learning & Statistical Learning
* Classification Techniques & Model Evaluation

## Repository Structure

```text
├── breast_cancer.csv        # Dataset file
├── classification_code.R   # R script for preprocessing and modeling
└── README.md               # Project documentation
```

## Future Work

* Explore ensemble learning methods such as Random Forest and XGBoost
* Apply cross-validation techniques for model assessment
* Investigate feature selection methods to improve model interpretability

## Author

**Hanieh Gholami Ghalhari**
MSc Student in Quantitative Finance and Insurance
University of Bergamo
