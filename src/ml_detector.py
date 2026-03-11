# src/ml_detector.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os

def train_and_detect():
    print("Loading Telemetry Data...")
    # Load from where Component 1 saved it
    data_path = os.path.expanduser("~/AMDProjects/gpu-failure-platform/data/telemetry_1m.csv")
    df = pd.read_csv(data_path)
    
    X = df[['voltage', 'temperature', 'memory_util']]
    y = df['failure_label']
    
    # 1. Supervised Learning (Known Failures)
    print("Training Random Forest Classifier...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    clf = RandomForestClassifier(n_estimators=100, n_jobs=-1) # Use all 16 CPUs
    clf.fit(X_train, y_train)
    
    print("Classification Report:")
    print(classification_report(y_test, clf.predict(X_test)))
    
    # Save Model
    joblib.dump(clf, "models/rf_gpu_model.pkl")
    
    # 2. Unsupervised Anomaly Detection (Unknown Issues)
    print("Training Isolation Forest for Anomalies...")
    iso = IsolationForest(contamination=0.01, n_jobs=-1)
    iso.fit(X)
    anomalies = iso.predict(X)
    
    print(f"Detected {sum(anomalies == -1)} anomalous events.")
    joblib.dump(iso, "models/iso_gpu_model.pkl")
    print("Models saved to models/")

if __name__ == "__main__":
    train_and_detect()
