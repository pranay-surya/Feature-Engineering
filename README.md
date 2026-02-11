# Feature Engineering with Python

This repository is a beginner-friendly and practical guide to **Feature Engineering**, one of the most important steps in any Machine Learning pipeline.
Feature engineering is the process of converting raw data into meaningful numerical features that machine learning models can understand and learn from.

---

##  Why Feature Engineering Matters

Most machine learning algorithms **cannot work directly with raw data** such as:
- Text values (`"Male"`, `"Female"`)
- Categories (`"Low"`, `"Medium"`, `"High"`)
- Missing values
- Different value scales (`age` vs `salary`)

Feature engineering helps:
- Improve model accuracy
- Reduce noise
- Make patterns easier to learn
- Prepare data correctly for ML algorithms

---

## What Is Feature Engineering?

**Feature engineering** is the art of transforming raw data into features that better represent the underlying problem to a machine learning model.

Common feature engineering tasks include:
- Encoding categorical variables
- Scaling numerical features
- Handling missing values
- Creating new features
- Removing unnecessary features

---

## What Is Label Encoding?

**Label Encoding** is a technique used to convert **categorical values (text labels)** into **numerical values**.

Most machine learning models only understand numbers, not words.

###  Example

Suppose you have a column:

| Color |
|------|
| Red |
| Blue |
| Green |

After **Label Encoding**, it becomes:

| Color |
|------|
| 0 |
| 1 |
| 2 |


###  When to Use Label Encoding
- For **ordinal data** (where order matters)
  - Example: `Low < Medium < High`
- For tree-based models (Decision Tree, Random Forest)

###  Important Note
Label Encoding can **introduce unintended order**.  
For example, the model might think `Green (2)` is greater than `Red (0)`, which may not be true.

---

## What Is One-Hot Encoding?

**One-Hot Encoding** converts each category into a **separate binary column**.

### Example

| Color | Red | Blue | Green |
|------|-----|------|-------|
| Red  | 1 | 0 | 0 |
| Blue | 0 | 1 | 0 |
| Green| 0 | 0 | 1 |

### When to Use One-Hot Encoding
- When categories have **no natural order**
- For linear models (Logistic Regression, Linear Regression)

---

## Train–Test Split

Before training a model, data is split into:

- **Training set** → used to train the model
- **Test set** → used to evaluate model performance

### Common Split Ratio
- 80% Training
- 20% Testing

This helps prevent **overfitting** and ensures the model performs well on unseen data.

---

## Feature Scaling

Feature scaling ensures that all numerical features are on a **similar scale**.

### Why is it needed?
Some algorithms are sensitive to large values.

Example:
- Age: `18 – 60`
- Salary: `20,000 – 1,000,000`

Without scaling, salary will dominate the model.

### Common Techniques
- **Standardization** (Mean = 0, Std = 1)
- **Normalization** (Range 0 to 1)

---

## Handling Missing Values

Real-world datasets often contain missing data.

This repository demonstrates techniques such as:
- Filling missing values using **mean / median / mode**
- Dropping rows or columns with missing values
- Choosing the right strategy based on data type

---






