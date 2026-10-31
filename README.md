# E-Commerce Delivery Delay Prediction

## Programming for Data Science – DPD601

### Project Overview

This project analyzes the Brazilian E-Commerce Public Dataset by Olist to investigate delivery delays and develop machine learning models for predicting whether an order will arrive later than its estimated delivery date.

The project includes data inspection, cleaning, aggregation, feature engineering, exploratory data analysis, machine learning, model evaluation, and automated testing.

## 1. Dataset

The project uses seven related datasets:

1. Orders
2. Customers
3. Order Items
4. Order Payments
5. Products
6. Sellers
7. Product Category Translation

The original orders dataset contains 99,441 orders.

The final modeling dataset contains 96,394 eligible orders.

### Target Definition

The target variable is `is_late`:

- `0`: Delivered on or before the estimated delivery calendar date.
- `1`: Delivered after the estimated delivery calendar date.

Only delivered orders with valid required dates and consistent event chronology are included in the modeling dataset.

## 2. Data Cleaning and Preparation

The data preparation process includes:

- Inspecting dataset dimensions, data types, and missing values.
- Checking primary keys and relationships.
- Parsing order timestamps.
- Handling missing and untranslated product categories.
- Aggregating order items before merging.
- Aggregating payments before merging.
- Preventing duplicated order records.
- Preserving missing values when appropriate.
- Identifying invalid chronological records.
- Documenting cleaning decisions.

Processed datasets are saved in `data/processed/`.

## 3. Feature Engineering

The project creates 14 engineered features, including:

- Approval day, month, and hour.
- Planned delivery duration.
- Order total value.
- Average item price.
- Freight ratio.
- Multiple-seller indicator.
- Multiple-payment indicator.
- Weekend approval indicator.
- Product weight features.
- Seller and customer geographic features.

Feature definitions and availability assumptions are documented in:

`data/processed/feature_dictionary.csv`

`data/processed/feature_availability_audit.csv`

## 4. Exploratory Data Analysis

Eight analytical questions are investigated using visualizations, including:

1. Distribution of late and on-time deliveries.
2. Monthly delivery delay rates.
3. Delivery delay rates by customer state.
4. Delivery delays by order size.
5. Order value by delivery status.
6. Planned delivery duration by delivery status.
7. Delay rates by approval day and hour.
8. Delivery status by number of sellers.

The visualizations include bar charts, line charts, box plots, histograms, and a heatmap.

Figures are stored in `outputs/figures/`.

Supporting analytical tables are stored in `outputs/tables/`.

## 5. Machine Learning

Three prediction approaches were evaluated:

- Majority-class baseline.
- Logistic Regression.
- Random Forest.

### Temporal Data Split

Data is split chronologically by order approval time:

| Dataset    | Orders |
| ---------- | -----: |
| Training   | 67,475 |
| Validation | 14,459 |
| Test       | 14,460 |

The validation dataset is used for model selection and threshold tuning.

The test dataset is reserved for final evaluation.

### Leakage Prevention

The final selected model uses six approval-time candidate features:

- `approval_day_of_week`
- `approval_month`
- `approval_hour`
- `weekend_approval`
- `planned_delivery_days`
- `customer_state`

The final model excludes actual delivery timestamps and final order status.

An important assumption is that the estimated delivery date is available at order approval time. Historical data does not independently establish its availability timestamp.

Missing-value imputation, scaling, and categorical encoding are fitted using training data only.

## 6. Final Model

Selected model: Logistic Regression

Selected classification threshold: 0.65

Model selection was performed using validation data.

### Final Test Results

| Metric            | Result |
| ----------------- | -----: |
| Accuracy          | 0.8041 |
| Precision         | 0.0593 |
| Recall            | 0.2411 |
| F1 Score          | 0.0952 |
| Average Precision | 0.0826 |

Confusion matrix:

| Actual / Predicted | On Time |  Late |
| ------------------ | ------: | ----: |
| On Time            |  11,479 | 2,363 |
| Late               |     469 |   149 |

The model identifies some late deliveries but produces false alarms and misses many delayed orders.

The decrease in performance from validation to test suggests that temporal changes may affect model generalization.

These results should not be interpreted as evidence of causal relationships.

## 7. Error Analysis

The final model's errors are analyzed by:

- Customer state.
- Planned delivery duration.
- False positives.
- False negatives.

Error analysis outputs are stored in `outputs/tables/`.

## 8. Technologies

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Pytest

## 9. Installation

Python 3.13 was used during development.

Create and activate a virtual environment on Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

### Extracting the Original Dataset

The submission includes the original dataset archive at:

`data/archive.zip`

Before running the project, extract the CSV files using:

```powershell
python -m zipfile -e data/archive.zip data
```

This command extracts the original CSV files directly into the `data/` directory.

The dataset archive contains the seven required Olist tables, along with the original geolocation and reviews files.

Do not modify the raw CSV files. All cleaned, aggregated, and engineered datasets are generated by the numbered Python scripts and saved in `data/processed/`.

The original archive is retained for reproducibility.

## 10. Running the Project

Execute the numbered Python scripts sequentially, starting with:

```powershell
python 01_data_inspection.py
```

Continue through the data cleaning, feature engineering, exploratory analysis, and machine learning scripts in numeric order.

Some scripts generate intermediate files required by later scripts.

To reproduce the final analysis, the processed datasets must be generated before running model evaluation.

## 11. Automated Tests

Run all tests:

```powershell
python -m pytest tests -v
```

The latest local test run completed successfully:

21 passed.

The tests check order uniqueness, aggregation totals, target validity, temporal ordering, selected feature names, and final prediction consistency.

## 12. Limitations

- The dataset represents historical Brazilian e-commerce activity.
- Delivery behavior and class proportions change over time.
- Late deliveries are less frequent than on-time deliveries.
- Some historical feature availability timestamps cannot be independently verified.
- Predictive relationships should not be interpreted as causal effects.
- The final model has limited precision and recall.
- The reported test results describe the selected model on the held-out period, not guaranteed future performance.

## 13. Data Source and License

Dataset: Brazilian E-Commerce Public Dataset by Olist

Citation:
Olist, & Sionek, A. (2018). Brazilian E-Commerce Public Dataset by Olist. Kaggle.

Source:
https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce

License: Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0).

The dataset is used for academic and non-commercial purposes. The original raw data is preserved without modification.

All processed datasets and analytical outputs are generated programmatically.

The findings describe historical Brazilian e-commerce data and should not be directly generalized to other markets.

## 14. Team Contributions

Team member names, responsibilities, code contributions, and peer-review evidence should be added before submission.

## 15. Academic Purpose

This project was prepared as part of the Programming for Data Science (DPD601) coursework.
