# Task 4 — Supervised Classification / Regression Modeling & Tuning

## Overview

This task focuses on building, tuning, comparing, and evaluating multiple supervised machine learning classification models for customer churn prediction.

The objective is to develop a complete machine learning workflow that includes:

* Training multiple classification algorithms
* Hyperparameter optimization
* Stratified cross-validation
* Model performance evaluation
* Model comparison
* Champion model selection
* Confusion matrix analysis
* ROC-AUC analysis
* Model serialization using Joblib

The implementation is developed using **Python** and **Scikit-Learn**, with XGBoost used as an additional classification algorithm.

---

## Dataset

The dataset used for this task is:

`customer-churn-training-clean.csv`

The dataset contains customer-level information used to predict whether a customer will churn.

### Target Variable

* `churned` — Target variable indicating whether the customer churned.

### Input Features

* `tenure_months`
* `support_tickets`
* `monthly_spend_inr`
* `last_login_days`
* `plan_type`

### Identifier

* `customer_id`

The `customer_id` column is excluded from model training because it is an identifier rather than a predictive feature.

---

## Project Structure

This task is part of the existing ML project repository:

```text
ML-Problem-Framing-Churn/
│
├── README.md
│
├── Task_1_ML_Problem_Framing/
│   └── ...
│
├── Task_3_Feature_Engineering/
│   └── ...
│
└── Task_4_Modeling_Tuning/
    ├── classification_modeling_tuning.ipynb
    ├── customer-churn-training-clean.csv
    ├── champion_model.joblib
    └── README.md
```

---

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-Learn
* XGBoost
* Joblib
* Jupyter Notebook / Google Colab

---

## Machine Learning Models

Four different classification architectures are trained and compared:

### 1. Logistic Regression

Logistic Regression is used as a linear baseline classification model.

### 2. Random Forest

Random Forest is an ensemble learning algorithm that combines multiple decision trees to improve predictive performance.

### 3. XGBoost

XGBoost is a gradient boosting algorithm used for powerful and efficient classification.

### 4. Support Vector Machine

SVM is used to identify a decision boundary that separates different classes.

---

## Data Preprocessing

The preprocessing workflow is implemented using Scikit-Learn Pipelines and `ColumnTransformer`.

### Numerical Features

Numerical features are processed using:

1. Median imputation for missing values
2. Standard scaling using `StandardScaler`

### Categorical Features

The categorical `plan_type` feature is processed using:

1. Most-frequent-value imputation
2. One-Hot Encoding

The preprocessing steps are included inside the machine learning pipelines so that transformations are learned only from the training data.

---

## Train-Test Split

The dataset is divided into training and testing sets before model preprocessing.

A stratified split is used to preserve the class distribution:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

This helps prevent information from the test set from being used during preprocessing or model training.

---

## Hyperparameter Optimization

`GridSearchCV` is used to find suitable hyperparameters for each model.

A **Stratified K-Fold Cross-Validation** strategy with 3 folds is used:

```python
cv = StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=42
)
```

The models are optimized using ROC-AUC as the primary scoring metric.

Example:

```python
grid = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="roc_auc",
    cv=cv,
    n_jobs=-1
)
```

---

## Evaluation Metrics

Each tuned model is evaluated using the following classification metrics:

### Precision

Measures how many predicted positive cases were actually positive.

### Recall

Measures how many actual positive cases were correctly identified.

### F1-Score

The harmonic mean of precision and recall.

### ROC-AUC

Measures the model's ability to distinguish between the two classes across classification thresholds.

### Confusion Matrix

Shows:

* True Positives
* True Negatives
* False Positives
* False Negatives

---

## Model Comparison

The notebook generates a model comparison table containing:

| Model               |             Precision |                Recall |              F1-Score |               ROC-AUC |
| ------------------- | --------------------: | --------------------: | --------------------: | --------------------: |
| Logistic Regression | Evaluated in notebook | Evaluated in notebook | Evaluated in notebook | Evaluated in notebook |
| Random Forest       | Evaluated in notebook | Evaluated in notebook | Evaluated in notebook | Evaluated in notebook |
| XGBoost             | Evaluated in notebook | Evaluated in notebook | Evaluated in notebook | Evaluated in notebook |
| SVM                 | Evaluated in notebook | Evaluated in notebook | Evaluated in notebook | Evaluated in notebook |

The actual values are generated when the notebook is executed.

---

## Champion Model

The champion model is selected based on the highest validation ROC-AUC performance from the evaluated models.

The selected model is stored in:

```text
champion_model.joblib
```

The model is serialized using Joblib:

```python
joblib.dump(
    champion_model,
    "champion_model.joblib"
)
```

The saved model can later be loaded without retraining:

```python
loaded_model = joblib.load(
    "champion_model.joblib"
)
```

Predictions can then be generated using:

```python
predictions = loaded_model.predict(X_test)
```

---

## Visualizations

The notebook includes the following visualizations:

### Confusion Matrix

The confusion matrix is generated for the selected champion model to analyze classification errors.

### ROC-AUC Curves

ROC curves for all four models are plotted together to compare their classification performance.

---

## Reproducibility

Random states are fixed where appropriate to make the experiments reproducible.

Example:

```python
random_state=42
```

The complete preprocessing and modeling workflow is implemented using Scikit-Learn Pipelines.

---

## Dataset Limitation

This dataset contains only **12 observations**. Therefore, model performance metrics and hyperparameter-selection results are highly sensitive to the train-test split and should not be interpreted as evidence of real-world generalization.

The purpose of this implementation is to demonstrate the complete classification modeling, hyperparameter tuning, evaluation, model comparison, and model serialization workflow.

A larger dataset would be required for reliable production-level model evaluation.

---

## Expected Deliverables

The completed task contains:

* `classification_modeling_tuning.ipynb` — Complete modeling and tuning notebook
* `customer-churn-training-clean.csv` — Dataset used for the experiment
* `champion_model.joblib` — Serialized trained champion model
* `README.md` — Documentation for Task 4

---

## Conclusion

This task demonstrates an end-to-end supervised classification workflow for customer churn prediction.

The workflow includes:

```text
Dataset
   ↓
Train-Test Split
   ↓
Preprocessing Pipeline
   ↓
Multiple Classification Models
   ↓
Hyperparameter Tuning
   ↓
Stratified Cross-Validation
   ↓
Model Evaluation
   ↓
Model Comparison
   ↓
Champion Model Selection
   ↓
Model Serialization
```

The resulting workflow provides a reusable foundation for evaluating supervised classification models while keeping preprocessing and model training organized within Scikit-Learn pipelines.
