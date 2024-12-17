# MLOPS and its application in Elevator Predictive Maintenace



## Project Description
This Project automates the predictive maintenace pipeline for an elevator predictive maintenance project, from data collection to model training and deployment, using MLflow for experiment tracking and deployment.

---

## Project Directory Structure

    Elevator-Predictive-Maintenance/ 
    ├── data/raw and pickle data                #folder of the datsets 
            ├── loaded_data.pkl                 #data already loaded and saved in a pickle file for easy reusalbility
            ├── file_names.pkl                  #file names saved for reusability
            ├── filtered_normalized_data.pkl    #data preprocessed saved in pickle file
            ├── journeys.pkl                    #index journeys saved for further use
            ├── raw_journey_data.pkl            #raw data within the lift range
            ├── all_cleaned_data.pkl            #filtered and normalized data + synthetic ready for feature extraction
            ├── extracted_features.csv          #csv file containing all the extracted features
    ├── Scripts/                                # Modular Python scripts 
        ├── Config                              
            ├── config.env                      #Configuration file
        ├── Load-raw-data/                      #Load raw data
            ├── load_data.py                    #Scripts to load raw data and save in pickle file
            ├── plot_raw_data.py                #Scripts to plot raw data saved in pickle file
            ├── utils.py                        #used for natural sort key             
        ├── Preprocessing/                      #Process the data(filter, cleaning)
            ├── data_cleaning.py                #data filtered and normalized, saved in pickle file
            ├── lift_journey_detection.py       #detect the elevator lift journey
            ├── plot_processed_journey_data.py  #plot filtered and normalized data with start and end journey identified
            ├── raw_journey_dataframe.py        #Update the dataframe focusing on the lift range
            ├── plot_raw_journey_dataframe.py   #plot raw data within the lift range
            ├── synthetic_data.py      #generate synthetic data using the montecarlo function to have more data
            ├── plot_raw_combined_data.py       #plot the updated raw datasets within lift journey original + synthetic
            ├── clean_combined_data.py          #filter and normalize the combined original + synthetic data
            ├── plot_clean_combined_data.py     #plot of all the data preprocessed within the lift journey combined with synthetic data
        ├── Calculated_features/ 
            ├── feature_extraction.py           #file of the feature extraction 
    ├── models/                                 # Trained models and MLflow artifacts 
            ├── neural_network.py               #neural_network model with performance tracking using mlfow
            ├── random_forest.py                #Random Forest model with performance tracking using mlflow
    ├── mlruns/                                 # MLflow tracking files 
    ├── main.py                                 # Main pipeline orchestrating script 
    ├── requirements.txt                        # Python dependencies 
    ├── README.md                               # Documentation

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
    python -m mlflow ui
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




















