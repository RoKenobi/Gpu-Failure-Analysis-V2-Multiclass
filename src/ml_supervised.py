# src/ml_supervised.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os

def train_supervised():
    print("Loading Telemetry Data for Supervised Learning...")
    data_path = os.path.expanduser("~/AMDProjects/gpu-failure-platform/data/telemetry_1m.csv")
    df = pd.read_csv(data_path)
    
    X = df[['voltage', 'temperature', 'memory_util']]
    y = df['failure_label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Map for human-readable output based on the new CUDA simulator logic
    target_names = ['Stable', 'Power Instability', 'Thermal Throttling', 'Memory Leak']
    
    # 1. Random Forest Baseline
    print("\n--- Training Random Forest Baseline ---")
    rf_clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    rf_clf.fit(X_train, y_train)
    print("Random Forest Results:")
    print(classification_report(y_test, rf_clf.predict(X_test), target_names=target_names))
    joblib.dump(rf_clf, "models/rf_gpu_model.pkl")
    
    # 2. XGBoost (Industry Standard)
    print("\n--- Training XGBoost Classifier ---")
    # XGBoost is highly optimized for multi-core HPC environments
    xgb_clf = xgb.XGBClassifier(n_estimators=100, tree_method='hist', n_jobs=-1, random_state=42)
    xgb_clf.fit(X_train, y_train)
    print("XGBoost Results:")
    print(classification_report(y_test, xgb_clf.predict(X_test), target_names=target_names))
    joblib.dump(xgb_clf, "models/xgb_gpu_model.pkl")

if __name__ == "__main__":
    train_supervised()
