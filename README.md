# bankruptcy-prediction-with-shap

# Interpretable Bankruptcy Prediction using Machine Learning

This repository contains a Python project that uses machine learning models to predict corporate bankruptcy based on financial ratios. The core focus is on using SHAP (SHapley Additive exPlanations) to interpret the model's predictions, making the results transparent and actionable.

## Project Overview

The goal of this project is to build a reliable bankruptcy prediction model and, more importantly, to understand which financial indicators are the most significant drivers of its predictions. We compare a simple Decision Tree with a more robust Random Forest model and then dive deep into the Random Forest's decision-making process using SHAP.

This approach moves beyond a "black box" model to provide clear, interpretable insights that can inform business strategy and financial consulting.

## Key Features

* **Model Training:** Implements and compares Decision Tree and Random Forest classifiers.
* **Performance Evaluation:** Reports key metrics like Precision, Recall, and F1-Score, with a focus on correctly identifying bankrupt firms (Class 1).
* **Global Interpretation:** Uses both default feature importance and SHAP values to identify critical financial ratios.
* **Explainability:** Leverages both `TreeExplainer` and `KernelExplainer` from the SHAP library to ensure consistency in feature attributions.

## Dataset

The project uses the `bankruptcy.csv` dataset, which contains 24 financial ratios (features `R1` to `R24`) for a set of companies. The target variable is:
* `D = 0`: Non-bankrupt company
* `D = 1`: Bankrupt company

## Installation & Setup

To run this project, you'll need Python 3 and a few key libraries.

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YourUsername/interpretable-bankruptcy-prediction.git](https://github.com/YourUsername/interpretable-bankruptcy-prediction.git)
    cd interpretable-bankruptcy-prediction
    ```

2.  **Create a `requirements.txt` file** with the following content:
    ```
    pandas
    numpy
    scikit-learn
    matplotlib
    seaborn
    shap
    ipykernel
    jupyter
    ```

3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

## Key Findings & Model Insights

The Random Forest model significantly outperformed the Decision Tree, achieving an **accuracy of 85%** and a strong **F1-Score of 0.80 for the bankrupt class**.

The SHAP analysis provided a clear and consistent global interpretation, revealing that the model's predictions are primarily driven by the following financially meaningful ratios:

1.  **R21 (Assets / Debts):** The strongest indicator of solvency. A higher value strongly suggests a company is healthy.
2.  **R18 (Income + Depreciation / Debts):** A key measure of cash flow available to service debt.
3.  **R14 (Income / Assets):** Reflects the efficiency of asset utilization to generate profit (Return on Assets).

These features consistently ranked as the most important across different SHAP explainers, building confidence in the model's reliability and interpretability.

<img width="816" height="593" alt="image" src="https://github.com/user-attachments/assets/c1789139-4e4f-4768-bb67-fedab0aa7fb3" />

<img width="1134" height="551" alt="image" src="https://github.com/user-attachments/assets/f285cdb4-0696-4182-af33-aa52265a6b08" />

---
