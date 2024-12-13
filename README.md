# MLOPS and its application in Elevator Predictive Maintenace



## Project Description
This Project automates the predictive maintenace pipeline for an elevator predictive maintenance project, from data collection to model training and deployment, using MLflow for experiment tracking and deployment.

---

## Project Directory Structure

    Elevator-Predictive-Maintenance/ 
    ├── data/                             # folder of the datsets 
        ├──raw/electric_training          #raw datasets
    ├── scripts/                          # Modular Python scripts 
    ├── models/                           # Trained models and MLflow artifacts 
    ├── mlruns/                           # MLflow tracking files 
    ├── main.py                           # Main pipeline orchestrating script 
    ├── requirements.txt                  # Python dependencies 
    ├── README.md                         # Documentation

---

## Getting Started

### **1. Prerequisites**

Before Starting, ensure the following are installed:
- **Python 3.8 or higher**
- **Pip**
- **MLflow** (Will be installed as part of 'requirements.txt')

### **2. Setting Up the Project**

1. **Clone the repository**:
    ```bash
    git clone <repository_url>
    cd project/

2. **Create a virtual environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate

3. **Install dependencies**:
    ```bash
    pip install -r requirement.txt

3. **Verifiy Mlflow Installation**:
    ```bash
    mlflow --version

---

## Running the Pipeline

### **1. Add your raw data**:
- Place your raw datasets in the **data/raw/** directory

### **2. Run the main pipeline**:
    python main.py

### **3. Track exepriments in Mlflow**:
- Start the mlflow UI:
    ```bash
    mlflow ui
- Open http://127.0.0.1:5000 in your browser

---

## Model Deployement

### **1. Serve the Model**:
- Use the following command to deploy the trained model:
    ```bash
    mlflow models serve -m "models:/random_forest_model/1" --port 1234

### **2. Test the Model API**:
- Use the example Python script to send data to the model and get predictions
    ```bash
    import requests
    import json

    url = "http://127.0.0.1:1234/invocations"
    headers = {"Content-Type": "application/json"}
    payload = {
        "columns": ["feature_sum", "feature_mean"],
        "data": [[1.2, 3.4]]
    }

    response = requests.post(url, headers=headers, json=payload)
    print(response.json())
---

## How to Reproduce on Another Machine

1. Clone the repository
2. Install a virtual environment and dependencies **(requirement.txt)**
3. Place your raw data in **data/raw/.**
4. Run the pipeline:
    ```bash
    python main.py

---
## Project Details

### **Features** :
- **Data Preprocessing**: Standardizes raw datasets.
- **Feature Engineering**: Extracys meaningful features for prediction.
- **MLflow Integration**: Tracks all experiments, parameters, metrics, and artifacts.
- **Model Deployment**: Serve the trained model via a REST API.

## Tecnologies
- **Python**
- **MLflow**
- **scikit-learn**




















