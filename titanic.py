# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# 1. Import the dataset and explore basic info
# Seaborn has a built-in function to load the Titanic dataset
df = sns.load_dataset('titanic')

print("--- Initial Data Exploration ---")
print("First 5 rows of the dataset:")
print(df.head())
print("\nDataset Information:")
df.info()
print("\nMissing Values Count:")
print(df.isnull().sum())
print("\nStatistical Summary:")
print(df.describe())

# 2. Handle missing values
# For 'age', we will impute missing values with the median.
age_median = df['age'].median()
df['age'].fillna(age_median, inplace=True)

# For 'embarked', we will impute with the mode.
embarked_mode = df['embarked'].mode()[0]
df['embarked'].fillna(embarked_mode, inplace=True)

# The 'deck' column has too many missing values (over 77%), so we'll drop it.
# The 'embark_town' is redundant with 'embarked'.
df.drop(['deck', 'embark_town'], axis=1, inplace=True)

print("\n--- After Handling Missing Values ---")
print("Missing Values Count:")
print(df.isnull().sum())

# 3. Convert categorical features into numerical using encoding
# and 4. Normalize/standardize numerical features.
# We can do this efficiently using a pipeline.

# Define categorical and numerical features
categorical_features = ['sex', 'embarked', 'pclass', 'who', 'adult_male', 'alone']
numerical_features = ['age', 'fare']

# Create preprocessing pipelines for both numerical and categorical data
numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Create a preprocessor object using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough' # Keep other columns (like 'survived')
)

# Apply the preprocessing pipeline to the dataframe
# The output will be a NumPy array
df_prepared = preprocessor.fit_transform(df)

# For demonstration, let's convert it back to a DataFrame to inspect
# Get feature names after one-hot encoding
ohe_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
all_feature_names = numerical_features + list(ohe_feature_names) + ['survived', 'alive'] # Adjust remainder columns if needed

# Note: The column order might differ slightly based on remainder='passthrough'.
# For a real project, it's better to handle all columns explicitly.
df_prepared_inspect = pd.DataFrame(df_prepared, columns=all_feature_names)

print("\n--- After Encoding and Standardization (First 5 rows) ---")
print(df_prepared_inspect.head())


# 5. Visualize outliers using boxplots and remove them
print("\n--- Outlier Visualization and Handling ---")
plt.figure(figsize=(10, 6))
sns.boxplot(x=df['fare'])
plt.title('Boxplot of Fare to Detect Outliers')
plt.show()

# Removing outliers using the IQR method for the 'fare' column
Q1 = df['fare'].quantile(0.25)
Q3 = df['fare'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filter out the outliers
df_no_outliers = df[(df['fare'] >= lower_bound) & (df['fare'] <= upper_bound)]

print(f"\nOriginal DataFrame shape: {df.shape}")
print(f"DataFrame shape after removing 'fare' outliers: {df_no_outliers.shape}")

