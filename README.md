<div align="center">

# Feature Engineering with Python

[![Python](https://img.shields.io/badge/Python-3.7%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Contributions Welcome](https://img.shields.io/badge/Contributions-Welcome-brightgreen?style=for-the-badge)](#-contributing)

**A comprehensive, beginner-friendly guide to Feature Engineering — the most impactful step in any Machine Learning pipeline.**

Feature engineering is the process of transforming raw, messy data into meaningful numerical features that machine learning models can understand, learn from, and make accurate predictions with.

---

</div>


## Why Feature Engineering Matters

Most machine learning algorithms **cannot work directly with raw data**:

| ❌ Problem | Example |
|:-----------|:--------|
| Text values | `"Male"`, `"Female"` |
| Unordered categories | `"Low"`, `"Medium"`, `"High"` |
| Missing / null values | `NaN`, blank cells |
| Vastly different scales | `Age: 18–60` vs `Salary: 20,000–1,000,000` |

**Feature engineering bridges this gap**, resulting in:

| ✅ Benefit | Description |
|:-----------|:------------|
| Higher Accuracy | Better features lead to better predictions |
| Reduced Noise | Cleaner data means fewer distractions for the model |
| Easier Learning | Patterns become more visible to algorithms |
| Algorithm Compatibility | Data formatted correctly for any ML model |

 ---

##  What Is Feature Engineering?

**Feature engineering** is the art and science of transforming raw data into features that better represent the underlying problem, enabling a machine learning model to learn more effectively.

### Core Tasks at a Glance 

| Encode Categorical → Numerical | Scale Numerical Features |  Impute Missing Values | Select Relevant Features |
|----------------------------------|-----------------------------|--------------------------|-----------------------------|
| Convert categorical data into numerical form (e.g., One-Hot, Label Encoding) | Normalize or standardize numeric features | Handle missing data using mean, median, or mode | Choose important features to improve model performance |


| Task | What It Does |
|:-----|:-------------|
| **Encoding** | Converts text/categories → numbers |
| **Scaling** | Brings all features to a similar range |
| **Imputation** | Fills in missing or null values |
| **Feature Creation** | Engineers new columns from existing data |
| **Feature Selection** | Removes irrelevant or redundant columns |

---

##  Techniques Covered

### 1️⃣ Label Encoding
- Converts **categorical text labels** into **numerical values**.

Most ML models only understand numbers — Label Encoding assigns a unique integer to each category.

#### Before vs After Encoding

| BEFORE (Categorical) | AFTER (Numerical) |
|----------------------|-------------------|
| Red                  | 0                 |
| Blue                 | 1                 |
| Green                | 2                 |


#### ✅ When to Use

- **Ordinal data** where order matters
Low (0) < Medium (1) < High (2)

- **Tree-based models** — Decision Tree, Random Forest, XGBoost, LightGBM

#### ⚠️ Important Caveat

- Label Encoding can **introduce unintended ordinal relationships**.
- The model may incorrectly assume `Green (2) > Red (0)`, even when no such order exists.
- For **nominal** (unordered) categories, prefer [One-Hot Encoding](#2️⃣-one-hot-encoding) instead.

---

### 2️⃣ One-Hot Encoding
- Converts each category into a **separate binary (0/1) column**, eliminating false ordinal relationships.

#### Before vs After (One-Hot Encoding)

| BEFORE (Color) | Red | Blue | Green |
|----------------|-----|------|-------|
| Red            | 1   | 0    | 0     |
| Blue           | 0   | 1    | 0     |
| Green          | 0   | 0    | 1     |

#### ✅ When to Use

- **Nominal data** with no inherent order (e.g., colors, cities, countries)
- **Linear models** — Logistic Regression, Linear Regression, Neural Networks

#### ⚠️ Watch Out For

- **High cardinality** — If a column has 100+ categories, One-Hot Encoding creates 100+ columns.
  Consider **Target Encoding** or **Frequency Encoding** as alternatives.
- **Multicollinearity** — Use `drop_first=True` to drop one column and avoid redundancy.

---

### 3️⃣ Train–Test Split

- Splits the dataset into **two separate sets** to ensure unbiased model evaluation.
#### Full Dataset Split

| TRAINING SET (80%) | TEST SET (20%) |
|--------------------|----------------|
| Used to **train** the model | Used to **evaluate** model performance |


####  Common Split Ratios

| Ratio | Training | Testing | Best For |
|:------|:---------|:--------|:---------|
| **80 / 20** | 80% | 20% | Standard datasets |
| **70 / 30** | 70% | 30% | Smaller datasets |
| **90 / 10** | 90% | 10% | Very large datasets |

####  Best Practice

- **Always split your data BEFORE applying any transformations** (scaling, encoding, imputation on full data)
- to prevent **data leakage** — where information from the test set leaks into training.

---

### 4️⃣ Feature Scaling

- Ensures all numerical features are on a **similar scale** so no single feature dominates the model.

#### Why Is It Needed?

Without scaling, features with larger values **dominate** the model:
#### Feature Range Problem

| Feature | Range | Impact |
|---------|-------|--------|
| Age     | 18 – 60 | Small range |
| Salary  | 20,000 – 1,000,000 | Dominates the model ⚠️ |

#### Common Techniques

| Technique | Formula | Output Range | Best For |
|:----------|:--------|:-------------|:---------|
| **Standardization** | `(x − μ) / σ` | Mean = 0, Std = 1 | Features with outliers; gradient-based models |
| **Normalization** | `(x − min) / (max − min)` | 0 to 1 | Bounded ranges; distance-based models |

#### ✅ When Required

- **Distance-based** — KNN, K-Means, SVM
- **Gradient-based** — Neural Networks, Logistic Regression, Linear Regression

#### 🚫 When Not Needed

- **Tree-based models** — Decision Trees, Random Forest, XGBoost (inherently scale-invariant)

---

### 5️⃣ Handling Missing Values

> Real-world datasets are rarely complete. Choosing the **right strategy** is crucial.

####  Common Strategies

| Strategy | Method | Best For |
|:---------|:-------|:---------|
| **Mean Imputation** | Replace `NaN` with column mean | Numerical data, no outliers |
| **Median Imputation** | Replace `NaN` with column median | Numerical data, **with** outliers |
| **Mode Imputation** | Replace `NaN` with most frequent value | Categorical data |
| **Forward / Backward Fill** | Use previous or next row's value | Time-series data |
| **Drop Rows** | Remove rows with `NaN` | Very few missing rows (<5%) |
| **Drop Columns** | Remove entire column | Column has >50% missing values |
| **Advanced (KNN / Model-based)** | Predict missing values | Critical features with patterns |

####  Decision Guide


| Condition | Action |
|-----------|--------|
| Column not important | ❌ Drop the column |
| Missing < 30% | ✅ Impute (mean / median / mode) |
| Missing > 50% | ❌ Drop the column |



---

##  Quick Examples

### Label Encoding

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
colors = ['Red', 'Blue', 'Green', 'Red', 'Blue']
encoded = le.fit_transform(colors)

print(encoded)  # [2, 0, 1, 2, 0]
```
### One-hot Encoding
```python
import pandas as pd

df = pd.DataFrame({'Color': ['Red', 'Blue', 'Green']})
encoded = pd.get_dummies(df, columns=['Color'], dtype=int)

print(encoded)
#    Color_Blue  Color_Green  Color_Red
# 0           0            0          1
# 1           1            0          0
# 2           0            1          0
```
### Train–Test Split
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```
### Feature Scaling
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # Fit + transform on train
X_test_scaled  = scaler.transform(X_test)         # Only transform on test

```
### Handling Missing Values
```python

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='median')
X_train_filled = imputer.fit_transform(X_train)
X_test_filled  = imputer.transform(X_test)

```
