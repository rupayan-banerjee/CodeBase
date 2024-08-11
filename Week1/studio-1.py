import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset from the specified file path
dataset_path = 'data.csv'
df = pd.read_csv(dataset_path)

# Display the initial few rows of the dataset to understand its structure
print("Initial sample rows from the dataset:")
print(df.head())

# Check and display the number of rows and columns in the dataset
print(f"Total number of rows: {df.shape[0]}")
print(f"Total number of columns: {df.shape[1]}")

# Display the column names along with their data types
print("\nData types for each column:")
print(df.dtypes)

# Eliminate duplicate entries from the dataframe
df_cleaned = df.drop_duplicates()

# Verify the number of rows remaining after removing duplicates
print(f"Number of rows after deduplication: {df_cleaned.shape[0]}")

# Generate box plots for all numerical columns to identify outliers
numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns

for column in numeric_cols:
    plt.figure(figsize=(10, 6))
    plt.boxplot(df_cleaned[column], vert=False)
    plt.title(f'Boxplot of {column}')
    plt.show()


# Function to remove outliers using Inter Quartile Range (IQR) method
def remove_outliers(df, cols, factor=1.5):
    for col in cols:
        Q1 = np.percentile(df[col], 25)
        Q3 = np.percentile(df[col], 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR

        # Filter out rows with outliers
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df


# Apply the outlier removal function multiple times if necessary
for _ in range(2):
    df_cleaned = remove_outliers(df_cleaned, numeric_cols)

# Generate box plots again to visualize the data after outlier removal
for column in numeric_cols:
    plt.figure(figsize=(10, 6))
    plt.boxplot(df_cleaned[column], vert=False)
    plt.title(f'Boxplot of {column} after outlier removal')
    plt.show()

# Check for any missing values in the cleaned dataset
print("\nCount of missing values in each column after cleaning:")
print(df_cleaned.isnull().sum())

# Fill in missing values with the mean of each column
df_cleaned.fillna(df_cleaned.mean(), inplace=True)

# Verify that all missing values have been addressed
print("\nCount of missing values after filling:")
print(df_cleaned.isnull().sum())

# Specify the target and feature variables
output_var = 'PE'
features = ['AT', 'V', 'AP', 'RH']

print(f"\nTarget Variable: {output_var}")
print(f"Features: {features}")

# Univariate analysis: Plot histograms for each feature variable
for feature in features:
    plt.figure(figsize=(10, 6))
    sns.histplot(df_cleaned[feature], kde=True, bins=30)
    plt.title(f'Distribution of {feature}')
    plt.xlabel(f'{feature}')
    plt.ylabel('Frequency')
    plt.show()

# Display summary statistics for all columns
summary = df_cleaned.describe()
print("\nSummary Statistics:")
print(summary)

# Multivariate analysis: Create scatter plots for feature pairs
sns.pairplot(df_cleaned[features + [output_var]], diag_kind='kde', corner=True)

# Adjust the size and position of the overall plot
plt.gcf().set_size_inches(12, 10)
plt.suptitle("Multivariate Analysis: Scatterplot Matrix", y=1.02)

# Display the scatter plots
plt.show()

# Compute the absolute correlation matrix for the dataset
correlation_matrix = abs(df_cleaned.corr())

# Extract only the lower triangle of the correlation matrix
lower_triangle = np.tril(correlation_matrix, k=-1)

# Mask the upper triangle values in the heatmap
mask = lower_triangle == 0

# Plot a heatmap of the lower triangle of the correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(lower_triangle, center=0.5, cmap='coolwarm', annot=True, xticklabels=correlation_matrix.index,
            yticklabels=correlation_matrix.columns,
            cbar=True, linewidths=1, mask=mask)
plt.title('Correlation Heatmap')
plt.show()
