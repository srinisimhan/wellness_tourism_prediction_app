# Wellness Tourism Prediction - MLOps Pipeline

## Project Overview
This project implements a complete MLOps pipeline for predicting customer purchases of wellness tourism packages. The system automates data preprocessing, model training, experimentation tracking, and deployment using Hugging Face Spaces with CI/CD via GitHub Actions.

## Business Problem
"Visit with Us" travel company needs to identify potential customers for their new Wellness Tourism Package. The manual approach is inefficient and error-prone. This solution provides an automated, data-driven system to predict customer purchase likelihood with 94% ROC AUC accuracy.

## Dataset
The dataset contains 4,888 customer records with 20 features including:
- **Demographic information**: Age, Gender, Occupation, Monthly Income
- **Travel preferences**: CityTier, PreferredPropertyStar, Passport ownership
- **Engagement data**: DurationOfPitch, NumberOfFollowups, PitchSatisfactionScore
- **Target variable**: `ProdTaken` (0 = Not Purchased, 1 = Purchased)

## Project Architecture
```
wellness_tourism_prediction_app/
├── .github/workflows/
│   └── pipeline.yml                    # CI/CD Pipeline
├── tourism_project/
│   ├── data/                           # Dataset files
│   ├── model_building/
│   │   ├── data_register.py           # Dataset registration
│   │   ├── prep.py                    # Data preprocessing
│   │   └── train.py                   # Model training
│   ├── deployment/
│   │   ├── app.py                     # Streamlit application
│   │   ├── Dockerfile                 # Container config
│   │   └── requirements.txt           # Dependencies
│   ├── hosting/
│   │   └── hosting.py                 # Deployment script
│   └── artifacts/                     # Model artifacts
├── requirements.txt                    # Project dependencies
└── README.md                          # This file
```

## Live Deployments

### **Hugging Face Spaces**
- **Application**: [Wellness Tourism Prediction App](https://huggingface.co/spaces/simnid/Wellness-Tourism-Prediction)
- **Model**: [wellness-tourism-model](https://huggingface.co/simnid/wellness-tourism-model)
- **Dataset**: [wellness-tourism-dataset](https://huggingface.co/datasets/simnid/wellness-tourism-dataset)

### **MLflow Experiment Tracking**
- **Tracking Server**: MLflow UI with ngrok tunnel
- **Experiment**: `wellness-tourism-prod-experiment`

## Technical Implementation

### **1. Data Pipeline**
- **Data Registration**: Automated upload to Hugging Face Hub
- **Data Preparation**: 
  - Column removal (CustomerID, Unnamed: 0)
  - Data cleaning (Gender typo correction)
  - Feature engineering (PitchEfficiency = Duration × Satisfaction)
  - Label encoding for categorical variables
- **Data Storage**: Versioned datasets on Hugging Face

### **2. Machine Learning Model**
- **Algorithm**: XGBoost Classifier
- **Class Imbalance Handling**: `scale_pos_weight = n_neg/n_pos`
- **Hyperparameter Tuning**: GridSearchCV with 5-fold stratified cross-validation
- **Preprocessing Pipeline**:
  - StandardScaler for numerical features
  - OneHotEncoder for categorical features
- **Evaluation Metrics**:
  - Primary: ROC AUC (0.9414)
  - Secondary: PR AUC, Precision, Recall, F1-Score

### **3. MLOps Components**
- **Experiment Tracking**: MLflow for parameter and metric logging
- **Model Registry**: Hugging Face Model Hub
- **Containerization**: Docker support for reproducibility
- **CI/CD**: GitHub Actions with 5-stage pipeline

### **4. Deployment Stack**
- **Frontend**: Streamlit web application
- **Backend**: XGBoost model with sklearn preprocessing
- **Hosting**: Hugging Face Spaces
- **API**: Direct model loading from HF Hub

## Model Performance

| Metric | Value |
|--------|-------|
| **ROC AUC** | 0.9414 |
| **PR AUC** | 0.8344 |
| **Accuracy** | 0.8898 |
| **Precision (Class 1)** | 0.69 |
| **Recall (Class 1)** | 0.79 |
| **F1-Score (Class 1)** | 0.74 |
| **Best Parameters** | n_estimators: 200, max_depth: 7, learning_rate: 0.1 |

## CI/CD Pipeline

The GitHub Actions workflow (`.github/workflows/pipeline.yml`) automates:

1. **Register Dataset**: Upload raw data to Hugging Face
2. **Data Preparation**: Clean, split, and process data
3. **Model Training**: Train with hyperparameter tuning and MLflow tracking
4. **Deploy to HF Space**: Deploy Streamlit app to Hugging Face
5. **Run Tests**: Validate pipeline execution

### **Secrets Required**:
- `HF_TOKEN`: Hugging Face authentication token
- `MLFLOW_TRACKING_URI`: MLflow server URL (optional)

## Local Development

### **Prerequisites**
- Python 3.9+
- Git
- Hugging Face Account with write token

### **Setup**
```bash
# Clone repository
git clone https://github.com/srinisimhan/wellness_tourism_prediction_app.git
cd wellness_tourism_prediction_app

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export HF_TOKEN="your_huggingface_token"
export MLFLOW_TRACKING_URI="http://localhost:5000"
```

## Using the Application
- **Access the web app:** Visit the Hugging Face Space link
- **Input customer details:** Fill in demographic, travel preferences, and engagement data
- **Get predictions:** Click "Predict Purchase Probability"
- **View insights:** See probability scores and business recommendations

### Sample Prediction Inputs:
- **High-probability customer:** Age 30-45, CityTier 2, Passport=1, PitchDuration > 20min
- **Low-probability customer:** Age > 50, CityTier 1, No passport, Short pitch duration

## Monitoring & Maintenance
### MLflow Tracking
- Access MLflow UI to track experiment history
- Compare model versions and parameters
- Monitor training metrics over time

### Model Retraining
The CI/CD pipeline automatically retrains the model when:
- New data is added to the dataset
- Code changes are pushed to the main branch
- Scheduled retraining (can be configured)

## Troubleshooting
### Common Issues
1. HF_TOKEN errors: Ensure token has write permissions
2. MLflow connection issues: Check ngrok tunnel is active
3. Model loading failures: Verify model file exists in HF Hub
4. Streamlit app crashes: Check dependency versions

### Debug Steps
```bash
# Test Hugging Face connection
python -c "from huggingface_hub import HfApi; api = HfApi(); print('Connected')"

# Test model loading
python -c "from huggingface_hub import hf_hub_download; import joblib; print('HF accessible')"

# Test Streamlit locally
streamlit run tourism_project/deployment/app.py --server.port 8501
```

## References
- [Hugging Face Hub Documentation](https://huggingface.co/docs/hub/index)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
