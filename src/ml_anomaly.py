# src/ml_anomaly.py
import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib
import os

def detect_anomalies():
    print("Loading Telemetry Data for Unsupervised Anomaly Detection...")
    data_path = os.path.expanduser("~/AMDProjects/gpu-failure-platform/data/telemetry_1m.csv")
    df = pd.read_csv(data_path)
    
    # Drop the labels completely - Isolation Forest operates blindly!
    X = df[['voltage', 'temperature', 'memory_util']]
    
    print("\n--- Training Isolation Forest ---")
    # We set contamination to 0.15 to match the 15% combined failure rate of our CUDA simulator
    iso = IsolationForest(contamination=0.15, n_jobs=-1, random_state=42)
    iso.fit(X)
    anomalies = iso.predict(X)
    
    # Isolation forest returns 1 for inliers (normal) and -1 for outliers (anomalies)
    anomaly_count = sum(anomalies == -1)
    normal_count = sum(anomalies == 1)
    
    print(f"Analysis Complete:")
    print(f" - Normal Execution States Detected: {normal_count}")
    print(f" - Anomalous Hardware States Detected: {anomaly_count}")
    
    joblib.dump(iso, "models/iso_gpu_model.pkl")
    print("Anomaly detection model saved to models/iso_gpu_model.pkl")

if __name__ == "__main__":
    detect_anomalies()
