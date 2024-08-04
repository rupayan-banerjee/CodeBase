import pandas as pd

# Read the CSV file
df = pd.read_csv('data.csv', header=None)

# Get the number of rows and columns
rows, columns = df.shape

# Get the column names and their types
column_info = df.dtypes

print(f"Number of rows: {rows}")
print(f"Number of columns: {columns}")
print("Column names and their types:")
print(column_info)
