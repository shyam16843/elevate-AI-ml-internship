import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Load the downloaded Retail Product dataset
df = pd.read_csv('synthetic_dataset.csv')

# 1. Explore basic info
print(df.info())
print(df.isnull().sum())

# 2. Handle missing values
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df.select_dtypes(include=['object']).columns.tolist()

# Impute numerical columns with median
for col in num_cols:
    median_val = df[col].median()
    df[col] = df[col].fillna(median_val)

# Impute categorical columns with mode
for col in cat_cols:
    mode_val = df[col].mode()[0]
    df[col] = df[col].fillna(mode_val)


# 3. Convert categorical features into numerical using one-hot encoding
df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# 4. Normalize/standardize numerical features
scaler = StandardScaler()
df_encoded[num_cols] = scaler.fit_transform(df_encoded[num_cols])

# 5. Visualize outliers using boxplots
plt.figure(figsize=(15, 5))
for i, col in enumerate(num_cols, 1):
    plt.subplot(1, len(num_cols), i)
    sns.boxplot(df_encoded[col])
    plt.title(f'Boxplot of {col}')
plt.suptitle('Boxplots of Numerical Features', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95]) 
plt.show()

# Remove outliers using IQR method
Q1 = df_encoded[num_cols].quantile(0.25)
Q3 = df_encoded[num_cols].quantile(0.75)
IQR = Q3 - Q1

condition = ~((df_encoded[num_cols] < (Q1 - 1.5 * IQR)) | (df_encoded[num_cols] > (Q3 + 1.5 * IQR))).any(axis=1)
df_clean = df_encoded[condition].reset_index(drop=True)

print("Shape after cleaning outliers:", df_clean.shape)
