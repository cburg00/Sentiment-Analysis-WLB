
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pickle

file_path = "spread sheet for DIB.csv"  
df = pd.read_excel(file_path, header=1)

# Clean column names
df.columns = [col.strip() if isinstance(col, str) else f"Unnamed_{i}" for i, col in enumerate(df.columns)]

# Select WLB columns
wlb_columns = [col for col in df.columns if "WLB" in col]

# Compute Average WLB Score
df["WLB_avg"] = df[wlb_columns].mean(axis=1)

# Create Sentiment Labels (1 = Positive, 0 = Negative)
df["sentiment"] = (df["WLB_avg"] >= 3.5).astype(int)

# Select Features and Target
X = df[wlb_columns]  
y = df["sentiment"]

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize Data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model Evaluation Function
def evaluate_model(y_true, y_pred, model_name):
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    return {"Model": model_name, "Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1-score": f1}

# Hyperparameter Tuning for Logistic Regression
log_reg = LogisticRegression()
log_param_grid = {'C': [0.001, 0.01, 0.1, 1, 10], 'solver': ['liblinear', 'lbfgs']}
log_grid = GridSearchCV(log_reg, log_param_grid, cv=5, scoring='accuracy', n_jobs=-1)
log_grid.fit(X_train_scaled, y_train)
best_log_reg = log_grid.best_estimator_
y_pred_log = best_log_reg.predict(X_test_scaled)

# Hyperparameter Tuning for Random Forest
rf_model = RandomForestClassifier(random_state=42)
rf_param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 10], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]}
rf_grid = GridSearchCV(rf_model, rf_param_grid, cv=5, scoring='accuracy', n_jobs=-1)
rf_grid.fit(X_train, y_train)
best_rf_model = rf_grid.best_estimator_
y_pred_rf = best_rf_model.predict(X_test)

log_reg_results = evaluate_model(y_test, y_pred_log, "Logistic Regression")
rf_results = evaluate_model(y_test, y_pred_rf, "Random Forest")

# Display Results
results_df = pd.DataFrame([log_reg_results, rf_results])
print("\nOptimized Model Evaluation Results:\n", results_df)

# Save optimized models
with open("optimized_logistic_regression.pkl", "wb") as f:
    pickle.dump(best_log_reg, f)

with open("optimized_random_forest.pkl", "wb") as f:
    pickle.dump(best_rf_model, f)
