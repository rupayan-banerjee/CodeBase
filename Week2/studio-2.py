import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Load the dataset from the CSV file
file = 'data.csv'
data = pd.read_csv(file)


# Define a function to categorize the 'PE' values into different labels
def categorize_pe(value):
    if value < 430:
        return 1
    elif 430 <= value < 450:
        return 2
    elif 450 <= value < 470:
        return 3
    elif 470 <= value < 490:
        return 4
    else:
        return 5


# Apply the categorization function to the 'PE' column
data['PE_Label'] = data['PE'].apply(categorize_pe)

# Save the categorized data to a new CSV file
output_path = 'converted_data.csv'
data.to_csv(output_path, index=False)

# Print the first few rows of the data to verify the categorization
print(data.head())
print()

# Reload the categorized data from the CSV file
file = 'converted_data.csv'
data = pd.read_csv(file)

# Plot the distribution of the 'PE_Label' categories
class_distribution = data['PE_Label'].value_counts().sort_index()

plt.figure(figsize=(10, 6))
class_distribution.plot(kind='bar', color='skyblue')
plt.title('Distribution of Energy Output Classes (PE_Label)')
plt.xlabel('Class')
plt.ylabel('Frequency')
plt.xticks(rotation=0)
plt.show()

# Select features to be normalized
features_to_normalize = ['AT', 'V', 'AP', 'RH']

# Initialize the MinMaxScaler
scaler = MinMaxScaler()

# Apply MinMax scaling to the selected features
data[features_to_normalize] = scaler.fit_transform(data[features_to_normalize])

# Save the normalized dataset to a new CSV file
data.to_csv('normalised_data.csv', index=False)

# Print the first few rows of the normalized data to verify the changes
print(data.head())
print()

# Reload the normalized data from the CSV file
data = pd.read_csv('normalised_data.csv')

# Create new composite features by multiplying pairs of normalized values
data['AT_V'] = data['AT'] * data['V']
data['AP_RH'] = data['AP'] * data['RH']

# Save the dataset with the new composite features
data.to_csv('features_data.csv', index=False)

# Print the first few rows to verify the new features
print(data.head())
print()

# Reload the dataset that includes the composite features
data = pd.read_csv('features_data.csv')

# Select the relevant features for further analysis
selected_features = ['AT', 'V', 'AP', 'AT_V', 'AP_RH', 'PE_Label']
selected_data = data[selected_features]

# Save the selected features to a new CSV file
selected_data.to_csv('selected_features_data.csv', index=False)

# Print the first few rows to verify the selected features
print(selected_data.head())
print()

# Reload the original dataset before normalization or feature engineering
data = pd.read_csv('converted_data.csv')

# Select the original relevant features
selected_features = ['AT', 'V', 'AP', 'PE_Label']

# Create composite features based on the original data
data['AT_V'] = data['AT'] * data['V']
data['AP_RH'] = data['AP'] * data['RH']

# Combine the original features with the new composite features
composite_features = ['AT_V', 'AP_RH']
final_selected_features = selected_features[:-1] + composite_features + ['PE_Label']
selected_data = data[final_selected_features]

# Save the final selected data to a new CSV file
selected_data.to_csv('selected_converted_data.csv', index=False)

# Print the first few rows to verify the final selected data
print(selected_data.head())
print()

# Reload the original categorized dataset
data = pd.read_csv('converted_data.csv')

# Define features and target for the first model
X = data[['AT', 'V', 'AP', 'RH']]
y = data['PE_Label']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Create and train a Decision Tree classifier
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Predict the response for the test dataset
y_pred = clf.predict(X_test)

# Calculate and print the accuracy of the first model
accuracy_model_1 = metrics.accuracy_score(y_test, y_pred)
print("Model 1 Accuracy:", accuracy_model_1)
print()

# Reload the normalized dataset
data = pd.read_csv('normalised_data.csv')

# Define features and target for the second model
X = data[['AT', 'V', 'AP', 'RH']]
y = data['PE_Label']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Create and train a Decision Tree classifier
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Predict the response for the test dataset
y_pred = clf.predict(X_test)

# Calculate and print the accuracy of the second model
accuracy_model_2 = metrics.accuracy_score(y_test, y_pred)
print("Model 2 Accuracy:", accuracy_model_2)
print()

# Reload the dataset with composite features
data = pd.read_csv('features_data.csv')

# Define features and target for the third model
X = data[['AT', 'V', 'AP', 'RH', 'AT_V', 'AP_RH']]
y = data['PE_Label']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Create and train a Decision Tree classifier
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Predict the response for the test dataset
y_pred = clf.predict(X_test)

# Calculate and print the accuracy of the third model
accuracy_model_3 = metrics.accuracy_score(y_test, y_pred)
print("Model 3 Accuracy:", accuracy_model_3)
print()

# Reload the dataset with selected features and composite features
data = pd.read_csv('selected_features_data.csv')

# Define features and target for the fourth model
X = data[['AT', 'V', 'AP', 'AT_V', 'AP_RH']]
y = data['PE_Label']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Create and train a Decision Tree classifier
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Predict the response for the test dataset
y_pred = clf.predict(X_test)

# Calculate and print the accuracy of the fourth model
accuracy_model_4 = metrics.accuracy_score(y_test, y_pred)
print("Model 4 Accuracy:", accuracy_model_4)
print()

# Reload the final dataset with all selected features
data = pd.read_csv('selected_converted_data.csv')

# Define features and target for the fifth model
X = data.drop(columns=['PE_Label'])
y = data['PE_Label']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Create and train a Decision Tree classifier
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Predict the response for the test dataset
y_pred = clf.predict(X_test)

# Calculate and print the accuracy of the fifth model
accuracy_model_5 = metrics.accuracy_score(y_test, y_pred)
print("Model 5 Accuracy:", accuracy_model_5)
print()

# Print a summary of the accuracies of all five models
print(f"Model 1 (No normalization, no composite features) Accuracy: {accuracy_model_1 * 100:.2f}%")
print(f"Model 2 (Normalization, no composite features) Accuracy: {accuracy_model_2 * 100:.2f}%")
print(f"Model 3 (Normalization, composite features) Accuracy: {accuracy_model_3 * 100:.2f}%")
print(f"Model 4 (Selected features with normalization) Accuracy: {accuracy_model_4 * 100:.2f}%")
print(f"Model 5 (Selected features without normalization) Accuracy: {accuracy_model_5 * 100:.2f}%")
