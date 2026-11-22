# Bank Marketing -- Feature Engineering and Machine Learning

## Project Overview

This project explores the Bank Marketing dataset using exploratory data
analysis (EDA), data visualization, preprocessing, feature selection,
and Logistic Regression classification.

The main objective is to examine relationships between numerical and
categorical features, visualize patterns in the data, and evaluate how
feature selection affects a predictive model's performance.

## Dataset

-   **File:** `bank-full.csv`
-   **Records:** 45,211
-   **Features:** 16 input features and 1 target column
-   **Target:** `y` --- whether the client subscribed to a term deposit
    (`yes` or `no`)
-   **Input feature types:** 7 numerical columns and 9 categorical
    columns

The notebook reads the CSV using a semicolon (`;`) delimiter.

### Main columns

The dataset includes customer and campaign information such as:

-   `age`, `balance`, `day`, `duration`, `campaign`, `pdays`, `previous`
-   `job`, `marital`, `education`, `default`, `housing`, `loan`,
    `contact`, `month`, `poutcome`
-   `y` --- target variable

## Project Workflow

### 1. Data loading and initial exploration

-   Load `bank-full.csv` with Pandas.
-   Inspect dataset dimensions, data types, sample records, descriptive
    statistics, target distribution, and missing values.

### 2. Exploratory data analysis and visualization

-   **Numerical correlation heatmap:** visualize correlations among the
    seven numerical features.
-   **Job vs. subscription heatmap:** use a normalized cross-tabulation
    to compare term-deposit subscription proportions across job
    categories.
-   **Pair plot:** visualize pairwise relationships among `age`,
    `balance`, `duration`, `campaign`, and `previous`, with points
    distinguished by the target class. A reproducible sample of 1,000
    rows is used for this plot.

### 3. Data cleaning

-   Create a copy of the original dataset.
-   Strip leading and trailing spaces from categorical values.
-   Convert target values in `y` to lowercase.
-   Remove duplicate records.
-   Replace `unknown` values in `job` and `education` with their
    respective modes.
-   Retain `unknown` categories in other categorical columns, as in the
    notebook.

### 4. Preprocessing

-   Separate the input features from the target.
-   Encode the target as `no = 0` and `yes = 1`.
-   Split the data into training and testing sets using an 80:20 split,
    `random_state=42`, and stratification.
-   Scale numerical features using `MinMaxScaler`.
-   One-hot encode categorical features using
    `OneHotEncoder(handle_unknown="ignore")`.
-   Fit the preprocessing transformations on the training data and apply
    them to the test data.

### 5. Baseline Logistic Regression

Train a Logistic Regression classifier with: - `max_iter=1000` -
`class_weight="balanced"`

Evaluate the model using accuracy, precision, recall, F1-score, and
ROC-AUC.

### 6. Feature selection

-   Apply `SelectKBest` with the chi-square (`chi2`) score function.
-   Select the top 20 features from the 49 preprocessed features.
-   Train a second Logistic Regression model using the selected
    features.
-   Evaluate it using the same metrics.

### 7. Model comparison

Compare the baseline model and feature-selected model using a results
table and a bar chart.

## Results

### Baseline Logistic Regression

  Metric         Score
  ----------- --------
  Accuracy      0.8464
  Precision     0.4192
  Recall        0.8119
  F1-score      0.5529
  ROC-AUC       0.9074

### Logistic Regression after Feature Selection

  Metric         Score
  ----------- --------
  Accuracy      0.8452
  Precision     0.4153
  Recall        0.7930
  F1-score      0.5452
  ROC-AUC       0.9031

### Feature Selection Summary

-   Features before selection: **49**
-   Features retained: **20**
-   The chi-square feature selection step reduced the feature set by 29
    features.

The selected features were:

-   Numerical: `duration`, `pdays`
-   Categorical indicators: `job_blue-collar`, `job_retired`,
    `job_student`, `marital_single`, `education_tertiary`, `housing_no`,
    `housing_yes`, `loan_yes`, `contact_cellular`, `contact_unknown`,
    `month_apr`, `month_dec`, `month_mar`, `month_may`, `month_oct`,
    `month_sep`, `poutcome_success`, `poutcome_unknown`

### Key Finding

The baseline model achieved a ROC-AUC of **0.9074**, while the
feature-selected model achieved **0.9031**. The feature selection
process reduced the number of input features from 49 to 20, with a small
decrease across the reported evaluation metrics. In this experiment,
reducing the feature set did not improve predictive performance,
although it produced a more compact model.

## Technologies and Libraries

-   Python
-   Jupyter Notebook
-   Pandas
-   NumPy
-   Matplotlib
-   Seaborn
-   Scikit-learn

## Repository Structure

``` text
Bank-Marketing-Feature-Engineering-and-ML/
│
├── DV_LAB_3.ipynb
├── bank-full.csv
├── README.md
└── requirements.txt
```

## How to Run

1.  Clone or download this repository.

2.  Install the required Python packages:

    ``` bash
    pip install -r requirements.txt
    ```

3.  Open `DV_LAB_3.ipynb` in Jupyter Notebook, JupyterLab, or Google
    Colab.

4.  Ensure `bank-full.csv` is in the same working directory as the
    notebook.

5.  Run the notebook cells in order.

## Notes

-   The notebook expects the dataset filename to be `bank-full.csv` and
    uses `sep=";"` when reading it.
-   Model scores are the results recorded in this notebook and may vary
    if the data, preprocessing, split, or model settings are changed.
-   This is an educational analysis and classification exercise, not a
    production banking decision system.
