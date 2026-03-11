# GPU Failure Simulation & Root Cause Analysis Platform 🔍⚡

## Overview
Modern GPU failures rarely occur as isolated events. Hardware instability typically emerges through cascading subsystem interactions (e.g., Voltage Instability → Thermal Rise → Memory Pressure → GPU Crash). 

This project is an enterprise-scale GPU reliability analysis platform that parallel-simulates hardware telemetry on an NVIDIA A100 and uses a dual-pronged machine learning architecture to diagnose root causes and detect zero-day anomalies.

Deployed on the **NSCC ASPIRE2A High-Performance Computing (HPC) cluster**, the system generates and analyzes 1,000,000+ diagnostic records, proving the ability to handle data-center-scale telemetry pipelines.

---

## System Architecture

The pipeline consists of four decoupled microservices executed via PBS job scheduling:

1. **CUDA Telemetry Simulator:** Uses `numba.cuda` to parallel-generate 1 million synthetic GPU states. Injects a 15% combined failure rate featuring overlapping noise between Power Instability, Thermal Throttling, and Memory Leaks.
2. **Graph-Based Root Cause Analyzer:** Uses `networkx` to model subsystem propagation paths (e.g., Power Delivery → Voltage Reg → GPU Core) and calculates degree centrality to identify critical vulnerability nodes.
3. **Supervised ML Pipeline (Diagnostics):** Trains **XGBoost** and **Random Forest** classifiers on the labeled telemetry to diagnose specific failure modes despite overlapping physical symptoms.
4. **Unsupervised ML Pipeline (Zero-Day Detection):** Uses an **Isolation Forest** operating completely blindly (labels dropped) to successfully isolate anomalous execution states from healthy telemetry.

---

## Technical Stack
* **Hardware:** NVIDIA A100-SXM4-40GB 
* **HPC Environment:** PBS Pro Workload Manager, Linux/Bash
* **Parallel Computing:** CUDA, Numba (v0.56.4)
* **Machine Learning:** XGBoost, Scikit-Learn (Random Forest, Isolation Forest)
* **Graph Analytics:** NetworkX

*Note on HPC Dependency Engineering: To ensure compatibility between the Numba compiler (PTX generation) and the host physical NVIDIA drivers on the ASPIRE2A compute nodes, this project utilizes a strictly controlled Conda environment locking `cudatoolkit=11.8` and `numba=0.56.4` with hardcoded `$NUMBA_CUDA_NVVM` paths to bypass host-level PTX 8.4 vs 9.2 mismatch errors.*

---

## Execution Results

### 1. Hardware Vulnerability Ranking
The graph centrality analysis successfully modeled the hardware dependency chain, proving that failures bottleneck at the power regulation and core logic layers before triggering memory or thermal sensors:
    - GPU Core: 0.8000
    - Voltage Reg: 0.6000
    - Thermal Sensor: 0.4000
    - Power Delivery / Memory Controller / Driver: 0.2000

### 2. Diagnostic Performance (XGBoost & Random Forest)
By engineering realistic, overlapping failure states (e.g., a memory leak causes a slight thermal rise; thermal throttling causes voltage to drop), the models achieved highly realistic diagnostic performance on a 200,000-record test set:
    - Stable Operation: 1.00 F1-Score
    - Memory Leak: 1.00 F1-Score
    - Power Instability: 0.99 F1-Score (0.98 Recall)
    - Thermal Throttling: 0.99 F1-Score (0.98 Recall)
*The 0.98 recall on Power/Thermal indicates the models successfully learned to navigate the complex overlapping noise between voltage spikes and heat generation.*

### 3. Anomaly Detection (Isolation Forest)
The unsupervised pipeline scanned the 1,000,000 records without access to failure labels and accurately segmented the exact 15% failure rate injected by the CUDA simulator:
    - Normal Execution States Detected: 850,000
    - Anomalous Hardware States Detected: 150,000

---

## How to Run on an HPC Cluster

1. Create the strictly versioned Conda environment on the login node:
    conda create -n test_ai_amd python=3.10 numpy=1.23.5 pandas scikit-learn networkx numba=0.56.4 cudatoolkit=11.8 xgboost -c conda-forge -y

2. Submit the job via the PBS scheduler (ensuring the script exports the correct Conda NVVM paths):
    qsub run_job.pbs

3. Monitor the queue and review output logs:
    qstat -u $USER
    cat logs/pbs_output.txt
