import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load trained models
with open("optimized_logistic_regression.pkl", "rb") as f:
    log_reg_model = pickle.load(f)

with open("optimized_random_forest.pkl", "rb") as f:
    rf_model = pickle.load(f)

# Load the Excel file with the first row as headers
file_path = "/Users/srivalli_nalla/Downloads/DATASET/spread sheet for DIB.xlsx" 
df = pd.read_excel(file_path, header=1)

# Display the first few rows
print("First 5 rows of the dataset:")
print(df.head())

# Select WLB columns
wlb_columns = [col for col in df.columns if "WLB" in col]
print("\nWLB Columns:", wlb_columns)

# Compute average WLB score
df["WLB_avg"] = df[wlb_columns].mean(axis=1)

# Create sentiment labels (1 = Positive, 0 = Negative)
df["sentiment"] = (df["WLB_avg"] >= 3.5).astype(int)

# Display the first few rows with sentiment labels
print("\nFirst 5 rows with WLB average and sentiment labels:")
print(df[["WLB_avg", "sentiment"]].head())

# Extract features & labels
X = df[wlb_columns]
y = df["sentiment"]

# Standardize the features (only for Logistic Regression)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Predict using Logistic Regression
y_pred_log = log_reg_model.predict(X_scaled)

# Predict using Random Forest
y_pred_rf = rf_model.predict(X)

# Evaluation Function
def evaluate_model(y_true, y_pred, model_name):
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True)
    precision, recall, f1 = report['1']['precision'], report['1']['recall'], report['1']['f1-score']
    print(f"\n🔹 {model_name} Performance:")
    print(f"Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1-score: {f1:.2f}")
    return {"Model": model_name, "Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1-score": f1}

# Evaluate both models
log_reg_results = evaluate_model(y, y_pred_log, "Logistic Regression")
rf_results = evaluate_model(y, y_pred_rf, "Random Forest")

# Convert results into a DataFrame for easy viewing
results_df = pd.DataFrame([log_reg_results, rf_results])
print("\n Optimized Model Evaluation Results:\n", results_df)

# Confusion Matrix for Logistic Regression
cm = confusion_matrix(y, y_pred_log)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Logistic Regression - Confusion Matrix")
plt.show()

# Confusion Matrix for Random Forest
cm_rf = confusion_matrix(y, y_pred_rf)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Random Forest - Confusion Matrix")
plt.show()
