# AI & ML Internship - Task 1: Data Cleaning & Preprocessing

This repository contains the solution for Task 1, which focuses on cleaning and preparing the Titanic dataset for machine learning.

## Objective

The main objective of this task is to apply various data preprocessing techniques to a raw dataset to make it suitable for training a machine learning model.

## Dataset

The project uses the **Titanic dataset**, which is a classic dataset for binary classification. It was loaded directly using the Seaborn library.

## Preprocessing Steps Performed

1.  **Data Exploration**: Loaded the dataset and performed an initial analysis to understand its structure, data types, and identify missing values using `.info()`, `.describe()`, and `.isnull().sum()`.

2.  **Handling Missing Values**:
    *   Imputed the `age` column with its median value.
    *   Imputed the `embarked` column with its mode.
    *   Dropped the `deck` column due to a high percentage (>77%) of missing values.

3.  **Feature Encoding & Scaling**:
    *   **Categorical Features**: `sex` and `embarked` were converted into numerical format using **One-Hot Encoding**.
    *   **Numerical Features**: `age` and `fare` were scaled using **Standardization** (Z-score scaling) to bring them to a common scale with a mean of 0 and a standard deviation of 1.

4.  **Outlier Detection and Removal**:
    *   Visualized outliers in the `fare` column using a **Boxplot**.
    *   Applied the **IQR (Interquartile Range)** method to identify and filter out extreme outliers from the `fare` column.

## Tools and Libraries Used

*   **Python 3.x**
*   **Pandas**: For data manipulation and analysis.
*   **NumPy**: For numerical operations.
*   **Matplotlib & Seaborn**: For data visualization (specifically, boxplots).
*   **Scikit-learn**: For preprocessing tasks like `StandardScaler` and `OneHotEncoder`.

