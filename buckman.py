import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import numpy as np
import pandas as pd
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier


# Load the dataset
data = pd.read_excel("C:/Users/nithi/Downloads/Dataset_Buckman.xlsx")  # Replace "investment_dataset.xlsx" with your dataset filename

# Drop 'S. No.' column
data = data.drop(columns=['S. No.'])

# Separate features and target variable
X = data.drop(columns=['Risk Level', 'Return Earned'])  # Features
y = data['Return Earned']  # Target variable

# Convert categorical variables into numerical values
X_encoded = pd.get_dummies(X, drop_first=True)  # Drop first to avoid multicollinearity

# Train a Random Forest Classifier to predict Return Earned
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_encoded, y)

# Get feature importances
feature_importances = pd.Series(rf_classifier.feature_importances_, index=X_encoded.columns)

# Remove duplicate columns caused by one-hot encoding
feature_importances = feature_importances.groupby(feature_importances.index.str.split('_').str[0]).max()

# Sort feature importances in descending order and get top 7
top_features = feature_importances.sort_values(ascending=False).head(7)

# Display top contributing columns
print("Top 7 contributing columns for making the best investment decision:")
top = top_features.index.tolist()
top_features=list()
top_features.append(top[0])
top_features.append(top[4])
top_features.append(top[5])
top_features.append(top[6])
print(top_features)  # Print only the column headers


file_loc = "C:/Users/nithi/Downloads/Dataset_Buckman.xlsx"
dataset = pd.read_excel(file_loc)

# checking the distribution of Target Varibale
dataset['Return Earned'].value_counts()

X = dataset.drop(columns='Return Earned', axis=1)
Y = dataset['Return Earned']

X_encoded = pd.get_dummies(X)

X_train, X_test, y_train, y_test = train_test_split(X_encoded, Y, test_size=0.2, random_state=42)

print(X.shape, X_train.shape, X_test.shape)

# Initialize and fit the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Accuracy on training data
train_1_prediction = model.predict(X_train)
accuracy_on_training_data = accuracy_score(y_train, train_1_prediction)
print('Accuracy on training data = ', accuracy_on_training_data)

# Accuracy on test data
test_1_prediction = model.predict(X_test)
accuracy_on_test_data = accuracy_score(y_test, test_1_prediction)
print('Accuracy on test data = ', accuracy_on_test_data)

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
import numpy as np

# Define list of models
models = [
    RandomForestClassifier()
]

def compare_models_train_test(models,X_train, X_test, y_train, y_test ):
    for model in models:
        # Train the model
        model.fit(X_train, y_train)
        
        # Evaluate the model
        test_data_prediction = model.predict(X_test)
        print("Model:", model.__class__.__name__)
        
        # Compute and print accuracy, precision, and f1-score
        accuracy = accuracy_score(y_test, test_data_prediction)
        print('Accuracy score of the model:', accuracy)
        
        report = classification_report(y_test, test_data_prediction)
        print(report)
        print("\n")

# Example usage:
# compare_models_train_test(models, X_train, Y_train, X_test, Y_test)
compare_models_train_test(models, X_train, X_test, y_train, y_test)


# Load the dataset
data = pd.read_excel("C:/Users/nithi/Downloads/Dataset_Buckman.xlsx")  # Replace with your dataset filename

# Define features and target variables
X = data.drop(columns=['Return Earned', 'Risk Level'])  # Features
y_return = data['Return Earned']  # Target variable for return prediction
y_risk = data['Risk Level']  # Target variable for risk prediction

# Encode target variable 'Return Earned' using ordinal encoding
ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
y_return_encoded = ordinal_encoder.fit_transform(y_return.values.reshape(-1, 1))

# Convert categorical variables into numerical values
X_encoded = pd.get_dummies(X)

# Model Training
# Split the data into training and testing sets
X_train, X_test, y_return_train, y_return_test = train_test_split(X_encoded, y_return_encoded, test_size=0.2, random_state=42)
X_train, X_test, y_risk_train, y_risk_test = train_test_split(X_encoded, y_risk, test_size=0.2, random_state=42)

# Train Random Forest Regressor for predicting returns earned
return_model = RandomForestRegressor(n_estimators=100, random_state=42)
return_model.fit(X_train, y_return_train)

# Train Random Forest Classifier for predicting risk levels
risk_model = RandomForestClassifier(n_estimators=100, random_state=42)
risk_model.fit(X_train, y_risk_train)

# User Interface
def predict_return_and_risk():
    age = int(entry_age.get())
    role = entry_role.get()
    education = entry_education.get()
    marital_status = entry_marital_status.get()
    income = float(entry_income.get())
    percentage_of_investment = float(entry_percentage_of_investment.get())
    knowledge_about_investment = int(entry_knowledge_about_investment.get())

    # Make predictions
    features = pd.DataFrame({
        'Age': [age],
        'Education': [education],
        'Role': [role],
        'Marital Status': [marital_status],
        'Household Income': [income],
        'Percentage of Investment': [percentage_of_investment],
        'Knowledge level about investment': [knowledge_about_investment]
    })

    # One-hot encode categorical variables
    features = pd.get_dummies(features)

    # Ensure all columns are present
    missing_cols = set(X_encoded.columns) - set(features.columns)
    for col in missing_cols:
        features[col] = 0

    features = features[X_encoded.columns]  # Reorder columns to match the model's input order

    predicted_return_encoded = return_model.predict(features)[0]
    predicted_return = ordinal_encoder.inverse_transform([[int(predicted_return_encoded)]])[0]
    predicted_risk = risk_model.predict(features)[0]

    # Display result in a message box
    result_text = f"Predicted Return: {predicted_return}\nPredicted Risk Level: {predicted_risk}"
    messagebox.showinfo("Prediction Result", result_text)

# Create a tkinter window
window = tk.Tk()
window.title("Investment Prediction")
window.geometry("420x450")
window.configure(bg="#f0f0f0")

# Create style for the labels
style = ttk.Style()
style.configure("TLabel", background="#f0f0f0", foreground="#333333", font=("Helvetica", 10))

# Add input fields
label_heading_input = ttk.Label(window, text="Enter the details for the prediction", font=("Helvetica", 12, "bold"), background="#f0f0f0", foreground="#333333")
label_heading_input.grid(row=0, column=0, columnspan=2, padx=10, pady=10)

label_age = ttk.Label(window, text="Age:")
label_age.grid(row=1, column=0, padx=10, pady=10)
entry_age = ttk.Entry(window)
entry_age.grid(row=1, column=1, padx=10, pady=10)

label_role = ttk.Label(window, text="Role:")
label_role.grid(row=2, column=0, padx=10, pady=10)
entry_role = ttk.Entry(window)
entry_role.grid(row=2, column=1, padx=10, pady=10)

label_education = ttk.Label(window, text="Education:")
label_education.grid(row=3, column=0, padx=10, pady=10)
entry_education = ttk.Entry(window)
entry_education.grid(row=3, column=1, padx=10, pady=10)

label_marital_status = ttk.Label(window, text="Marital Status:")
label_marital_status.grid(row=4, column=0, padx=10, pady=10)
entry_marital_status = ttk.Entry(window)
entry_marital_status.grid(row=4, column=1, padx=10, pady=10)

label_income = ttk.Label(window, text="Household Income:")
label_income.grid(row=5, column=0, padx=10, pady=10)
entry_income = ttk.Entry(window)
entry_income.grid(row=5, column=1, padx=10, pady=10)

label_percentage_of_investment = ttk.Label(window, text="Percentage of Investment:")
label_percentage_of_investment.grid(row=6, column=0, padx=10, pady=10)
entry_percentage_of_investment = ttk.Entry(window)
entry_percentage_of_investment.grid(row=6, column=1, padx=10, pady=10)

label_knowledge_about_investment = ttk.Label(window, text="Knowledge level about investment (0-10):")
label_knowledge_about_investment.grid(row=7, column=0, padx=10, pady=10)
entry_knowledge_about_investment = ttk.Entry(window)
entry_knowledge_about_investment.grid(row=7, column=1, padx=10, pady=10)

# Add predict button
btn_predict = ttk.Button(window, text="Predict", command=predict_return_and_risk)
btn_predict.grid(row=8, column=0, columnspan=2, pady=10)

# Start the GUI
window.mainloop()
