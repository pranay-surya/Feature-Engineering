<div align="center">

# Feature Engineering with Python.

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

| Encode Categorical → Numerical | Scale Numerical Features | Impute Missing Values | Select Relevant Features |
|--------------------------------|--------------------------|----------------------|--------------------------|
| Convert categorical data into numerical form (e.g., One-Hot, Label Encoding) | Normalize or standardize numeric features | Handle missing data using mean, median, or mode | Choose important features to improve model performance |

| Handle DateTime Columns | Handle Mixed Variables | Binarization & Binning | Power & Math Transformations |
|-------------------------|------------------------|------------------------|------------------------------|
| Extract year, month, day, hour, weekday from raw timestamps | Separate columns containing both text and numeric data | Convert continuous values into binary flags or discrete buckets | Reduce skewness and engineer new features via mathematical operations |

| Task | What It Does | Common Tools |
|:-----|:-------------|:-------------|
| **Encoding** | Converts text/categories → numbers | `LabelEncoder`, `pd.get_dummies` |
| **Scaling** | Brings all features to a similar range | `StandardScaler`, `MinMaxScaler` |
| **Imputation** | Fills in missing or null values | `SimpleImputer`, `KNNImputer` |
| **DateTime Handling** | Extracts temporal features from timestamps | `pd.to_datetime`, `dt.accessor` |
| **Mixed Variables** | Splits columns with blended text + numbers | `str.extract`, regex |
| **Binarization** | Converts numeric values to 0/1 using a threshold | `Binarizer` |
| **Discretization** | Groups continuous values into discrete bins | `pd.cut`, `pd.qcut` |
| **Power Transformation** | Reduces skewness, normalizes distributions | `PowerTransformer`, `np.log1p` |
| **Math Transformation** | Creates new features via ratios, products, polynomials | `PolynomialFeatures`, pandas ops |
| **Feature Selection** | Removes irrelevant or redundant columns | `SelectKBest`, `feature_importances_` |

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
---

### 6️⃣ Handling Date & Time Features

- Extracts **meaningful numerical components** from raw datetime columns so models can learn temporal patterns.

Raw datetime strings like `"2024-03-15 14:30:00"` are useless to ML models. By decomposing them into year, month, day, hour, etc., we expose **seasonality, trends, and cyclic patterns**.

#### Before vs After (DateTime Extraction)

| BEFORE (Raw DateTime)     | AFTER (Extracted Features)                          |
|---------------------------|-----------------------------------------------------|
| `2024-03-15 14:30:00`     | year=2024, month=3, day=15, hour=14, weekday=4      |
| `2023-12-25 08:00:00`     | year=2023, month=12, day=25, hour=8, weekday=0      |

#### Common Features to Extract

| Feature       | Description                        | Example              |
|:--------------|:-----------------------------------|:---------------------|
| **Year**      | Calendar year                      | 2024                 |
| **Month**     | Month number (1–12)                | 3 (March)            |
| **Day**       | Day of month (1–31)                | 15                   |
| **Hour**      | Hour of day (0–23)                 | 14                   |
| **Weekday**   | Day of week (0=Mon, 6=Sun)         | 4 (Friday)           |
| **Quarter**   | Quarter of year (1–4)              | 1                    |
| **Is Weekend**| Binary flag for Sat/Sun            | 0 or 1               |
| **Days Since**| Elapsed days from a reference date | 365                  |

#### ✅ When to Use

- Any dataset with **timestamps, dates, or time-series** data
- E-commerce (purchase time), finance (transaction date), healthcare (admission date)

#### ⚠️ Watch Out For

- Always **parse datetime columns** with `pd.to_datetime()` before extraction — they are often stored as strings.
- For cyclic features like hour or month, consider **sine/cosine encoding** to preserve the circular nature (e.g., hour 23 is close to hour 0).

---

### 7️⃣ Mixed Variables

- Handles columns that contain **both numerical and categorical information** within the same field.

Mixed variables are common in real-world data — for example, `"A34"`, `"B12"`, `"99+"`. These must be **split and encoded separately** before any model can use them.

#### Before vs After (Mixed Variable)

| BEFORE (Mixed)  | Numeric Part | Categorical Part |
|-----------------|--------------|-----------------|
| `A34`           | 34           | A               |
| `B12`           | 12           | B               |
| `C99`           | 99           | C               |

#### Common Strategies

| Strategy              | Description                                          | Best For                          |
|:----------------------|:-----------------------------------------------------|:----------------------------------|
| **Split & Encode**    | Separate the text prefix and numeric suffix          | Alphanumeric codes                |
| **Regex Extraction**  | Use `str.extract()` with regex patterns              | Complex mixed formats             |
| **Map to Categories** | Group mixed values into logical buckets              | Low cardinality mixed columns     |

#### ✅ When to Use

- Columns like loan grades (`A1`, `B3`), product codes (`SKU-1042`), or age ranges (`25-34`)

#### ⚠️ Watch Out For

- Never feed raw mixed columns directly into a model — they will either **cause errors** or be silently misinterpreted.

---

### 8️⃣ Binarization

- Converts a **continuous numerical feature into a binary (0/1)** feature by applying a threshold.

Instead of using the raw value, binarization asks: *"Is this value above or below a threshold?"* — turning a regression-style input into a binary signal.

#### Before vs After (Binarization)

| BEFORE (Age) | Threshold (≥ 18) | AFTER (Is Adult) |
|--------------|------------------|-----------------|
| 15           | < 18             | 0               |
| 22           | ≥ 18             | 1               |
| 35           | ≥ 18             | 1               |
| 10           | < 18             | 0               |

#### ✅ When to Use

- When the **presence or absence** of a condition matters more than the exact value
- Binary classification tasks with natural cutoff points (age thresholds, test scores, temperatures)
- Simplifying noisy continuous features for **interpretable models**

#### 🚫 When Not to Use

- When the **magnitude** of the feature carries important information
- With **tree-based models** that can already find optimal splits on raw values

---

### 9️⃣ Discretization / Binning

- Groups **continuous values into discrete intervals (bins)**, converting a numerical feature into an ordinal or categorical one.

Rather than treating `Age = 23` and `Age = 27` as entirely different values, binning groups them both into a `"Young Adult"` bucket — reducing noise and helping models find broader patterns.

#### Before vs After (Binning)

| BEFORE (Age) | AFTER (Age Group)  |
|--------------|--------------------|
| 8            | Child (0–12)       |
| 17           | Teen (13–17)       |
| 25           | Young Adult (18–35)|
| 45           | Middle-Aged (36–60)|
| 70           | Senior (60+)       |

#### Common Binning Strategies

| Strategy             | Description                                    | Best For                         |
|:---------------------|:-----------------------------------------------|:---------------------------------|
| **Equal Width**      | Bins of equal size across the value range       | Uniformly distributed data       |
| **Equal Frequency**  | Each bin contains the same number of samples    | Skewed distributions             |
| **Custom / Domain**  | Bins defined by domain knowledge (e.g., age groups) | Interpretable business features |
| **KMeans Binning**   | Cluster-based bin boundaries                   | Complex, non-uniform patterns    |

#### ✅ When to Use

- Reducing the effect of **outliers** in numerical features
- Creating **interpretable features** for stakeholders
- When the relationship between a feature and target is **non-linear** or step-like

#### ⚠️ Watch Out For

- **Information loss** — binning discards within-bin variation.
- Choose bin boundaries carefully; poor choices can obscure real patterns.

---

### 🔟 Power Transformations

- Applies a **mathematical power function** to a numerical feature to reduce skewness and make distributions more **normal (Gaussian)**.

Many ML algorithms assume features are normally distributed. Skewed features — like income or house prices — can hurt model performance. Power transformations correct this.

#### Before vs After (Power Transformation)

| BEFORE (Skewed Income) | AFTER (Log Transformed) |
|------------------------|------------------------|
| 5,000                  | 8.52                   |
| 50,000                 | 10.82                  |
| 500,000                | 13.12                  |
| 5,000,000              | 15.42                  |

#### Common Techniques

| Technique               | Formula / Method                          | Best For                                      |
|:------------------------|:------------------------------------------|:----------------------------------------------|
| **Log Transform**       | `log(x)` or `log(x + 1)`                 | Right-skewed data; values > 0                 |
| **Square Root**         | `√x`                                      | Mildly right-skewed; count data               |
| **Box-Cox**             | Optimal λ found via MLE                   | Positive-only data; automatic skew correction |
| **Yeo-Johnson**         | Extended Box-Cox                          | Data with **zero or negative** values         |

#### ✅ When to Use

- Features with heavy **right or left skew** (income, price, population)
- **Linear models** and **distance-based models** that are sensitive to distribution shape
- When residuals from a model are not normally distributed

#### 🚫 When Not to Use

- **Tree-based models** — they are invariant to monotonic transformations (no benefit)
- When the feature is already approximately normal

---

### 1️⃣1️⃣ Mathematical Transformations

- Creates **new features** by applying mathematical operations to existing ones, uncovering hidden relationships that raw features may not capture.

Sometimes the interaction or ratio between two features is far more predictive than either feature alone. Mathematical transformations let you engineer these signals explicitly.

#### Common Transformations

| Transformation         | Formula                    | Example Use Case                          |
|:-----------------------|:---------------------------|:------------------------------------------|
| **Ratio**              | `A / B`                    | `Revenue / Employees` → Revenue per head  |
| **Difference**         | `A − B`                    | `Sale Price − Cost Price` → Profit        |
| **Product**            | `A × B`                    | `Hours Worked × Hourly Rate` → Earnings   |
| **Polynomial**         | `x²`, `x³`, `x₁ × x₂`     | Capture non-linear relationships          |
| **Reciprocal**         | `1 / x`                    | Speed from time; rate features            |
| **Absolute Value**     | `\|x\|`                    | Magnitude of change regardless of sign    |

#### Before vs After (Mathematical Transformation)

| BEFORE                          | Transformation       | AFTER (New Feature)      |
|---------------------------------|----------------------|--------------------------|
| Revenue = 500,000 / Staff = 50  | Ratio                | Revenue per Staff = 10,000 |
| Sale = 1,200 / Cost = 800       | Difference           | Profit = 400             |
| Hours = 40 / Rate = 25          | Product              | Earnings = 1,000         |

#### ✅ When to Use

- When **domain knowledge** suggests a relationship between two features
- To explicitly encode **interaction effects** for linear models
- Polynomial features for capturing **non-linearity** without switching to a complex model

#### ⚠️ Watch Out For

- **Division by zero** — always add a small epsilon (`+ 1e-9`) when dividing
- Creating too many polynomial features can cause **overfitting** and the curse of dimensionality
- Always check that new features **improve model performance** via cross-validation

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

### Handling Date & Time Features
```python
import pandas as pd

df['date'] = pd.to_datetime(df['date'])

df['year']       = df['date'].dt.year
df['month']      = df['date'].dt.month
df['day']        = df['date'].dt.day
df['hour']       = df['date'].dt.hour
df['weekday']    = df['date'].dt.weekday
df['is_weekend'] = df['date'].dt.weekday.isin([5, 6]).astype(int)
```

### Mixed Variables
```python
import pandas as pd

df['code'] = ['A34', 'B12', 'C99']

df['code_letter'] = df['code'].str.extract(r'([A-Za-z]+)')
df['code_number'] = df['code'].str.extract(r'(\d+)').astype(int)
```

### Binarization
```python
from sklearn.preprocessing import Binarizer

binarizer = Binarizer(threshold=18)
df['is_adult'] = binarizer.fit_transform(df[['age']])
```

### Discretization / Binning
```python
import pandas as pd

bins   = [0, 12, 17, 35, 60, 100]
labels = ['Child', 'Teen', 'Young Adult', 'Middle-Aged', 'Senior']

df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels)
```

### Power Transformations
```python
from sklearn.preprocessing import PowerTransformer

pt = PowerTransformer(method='yeo-johnson')
X_train_transformed = pt.fit_transform(X_train)
X_test_transformed  = pt.transform(X_test)
```

### Mathematical Transformations
```python
import pandas as pd
import numpy as np

# Ratio
df['revenue_per_employee'] = df['revenue'] / (df['employees'] + 1e-9)

# Difference
df['profit'] = df['sale_price'] - df['cost_price']

# Polynomial
df['age_squared'] = df['age'] ** 2

# Log transform
df['log_income'] = np.log1p(df['income'])  # log(1 + x) handles zeros safely
```
