import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report
)
import mlflow
import mlflow.sklearn
import os
from dotenv import load_dotenv

dotenv_path = os.path.join(os.path.dirname(__file__), "../Config/config.env")
load_dotenv(dotenv_path)

# Load the extracted features dataset
features_file_path = os.getenv("OUTPUT_FEATURES_PATH", "../data/extracted_features.csv")

print(f"Loaded OUTPUT_FEATURES_PATH: {features_file_path}")

if features_file_path and os.path.exists(features_file_path):
    dataFrame_features = pd.read_csv(features_file_path)
else:
    raise FileNotFoundError(f"File not found at path: {features_file_path}")

# ========================== Preprocessing Phase ==========================
# Split data into features and labels
X = dataFrame_features.drop(columns=["label"])
y = dataFrame_features["label"]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA to retain 95% of the variance
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)

# Train-test split after PCA
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# ========================== Model Training Phase ==========================
# Initialize and train the model
rf_model = RandomForestClassifier(
    n_estimators=200, 
    max_depth=None, 
    random_state=42
)
rf_model.fit(X_train, y_train)

# Predictions and probabilities
y_pred = rf_model.predict(X_test)
y_pred_proba = rf_model.predict_proba(X_test)[:, 1]

# ========================== Evaluation Metrics ==========================
# Calculate performance metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"ROC-AUC: {roc_auc}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Cross-validation
cv_scores = cross_val_score(rf_model, X_scaled, y, cv=5, scoring='accuracy')
print(f"Cross-Validation Scores: {cv_scores}")
print(f"Mean CV Accuracy: {cv_scores.mean()}")

# ========================== MLflow Tracking Phase ==========================
# Initialize and log the experiment with MLflow
mlflow.set_experiment("Random Forest Experiment")

with mlflow.start_run(run_name="Random Forest"):
    # Log parameters
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", None)
    mlflow.log_param("random_state", 42)
    
    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("roc_auc", roc_auc)
    mlflow.log_metric("mean_cv_accuracy", cv_scores.mean())
    
    # Log model
    mlflow.sklearn.log_model(rf_model, "random_forest_model")
    print("Random Forest experiment logged successfully!")