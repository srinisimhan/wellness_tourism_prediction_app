# for data manipulation
import pandas as pd
import numpy as np
# for model training, tuning, and evaluation
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, recall_score, roc_auc_score, precision_recall_curve, auc
# for model serialization
import joblib
# for creating a folder
import os
# for hugging face space authentication to upload files
from huggingface_hub import HfApi, create_repo, hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError
import mlflow

# Get tokens from environment
HF_TOKEN = os.getenv("HF_TOKEN")
# Strip any potential whitespace from the token
if HF_TOKEN:
    HF_TOKEN = HF_TOKEN.strip()
if not HF_TOKEN:
    raise EnvironmentError("HF_TOKEN environment variable not set!")

# Hugging Face repositories
DATA_REPO_ID = "simnid/wellness-tourism-dataset"
MODEL_REPO_ID = "simnid/wellness-tourism-model"

# mlflow setup
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("wellness-tourism-prod-experiment")

# inititalizing client
api = HfApi(token=HF_TOKEN)

# Define local directory for downloaded data
local_data_dir = "tourism_project/data_downloaded"
os.makedirs(local_data_dir, exist_ok=True)

# Function to download and load data
def download_and_load_csv(filename, repo_id, local_dir, repo_type="dataset"):
    local_file_path = os.path.join(local_dir, filename)
    if not os.path.exists(local_file_path):
        print(f"Downloading {filename} from {repo_id}...")
        hf_hub_download(repo_id=repo_id, filename=filename, local_dir=local_dir, repo_type=repo_type, token=HF_TOKEN)
    return pd.read_csv(local_file_path)

# Loading datasets from Hugging Face dataset
Xtrain = download_and_load_csv("Xtrain.csv", DATA_REPO_ID, local_data_dir)
Xtest = download_and_load_csv("Xtest.csv", DATA_REPO_ID, local_data_dir)
ytrain = download_and_load_csv("ytrain.csv", DATA_REPO_ID, local_data_dir).squeeze()
ytest = download_and_load_csv("ytest.csv", DATA_REPO_ID, local_data_dir).squeeze()

print("Data loaded successfully from Hugging Face")
print(f"Train shapes: X={Xtrain.shape}, y={ytrain.shape}")
print(f"Test shapes: X={Xtest.shape}, y={ytest.shape}")

# Processing the features
# Identifing column types
num_cols = Xtrain.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = Xtrain.select_dtypes(exclude=[np.number]).columns.tolist()

# Move specific numeric columns to categorical
for col in ["CityTier", "PreferredPropertyStar", "Passport", "OwnCar"]:
    if col in num_cols:
        num_cols.remove(col)
        if col not in cat_cols:
            cat_cols.append(col)

print(f"Numeric columns ({len(num_cols)}): {num_cols}")
print(f"Categorical columns ({len(cat_cols)}): {cat_cols}")

# Setting up the preprocessing pipeline
preprocessor = make_column_transformer(
    (StandardScaler(), num_cols),
    (OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
)

# Class Imbalance handling
# Calculate scale_pos_weight for XGBoost
n_pos = int(ytrain.sum())
n_neg = len(ytrain) - n_pos
scale_pos_weight = n_neg / (n_pos + 1e-6)
print(f"Class weights - Positive: {n_pos}, Negative: {n_neg}, scale_pos_weight: {scale_pos_weight:.2f}")

# Model definition
xgb_model = xgb.XGBClassifier(
    use_label_encoder=False,
    eval_metric="logloss",
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1
)

model_pipeline = make_pipeline(preprocessor, xgb_model)

# Parameter Grid - Reduced.
param_grid = {
    'xgbclassifier__n_estimators': [100, 200],           # Keep 2 (most important)
    'xgbclassifier__max_depth': [5, 7],                  # Keep 2 (3,5,7 -> 5,7)
    'xgbclassifier__learning_rate': [0.01, 0.05, 0.1],   # Keep 3 (very important)
    'xgbclassifier__colsample_bytree': [0.8],            # Reduce to 1 (0.8 usually best)
    'xgbclassifier__reg_lambda': [1.0]                   # Reduce to 1 (1.0 is default)
}

# Starting MLflow run
with mlflow.start_run():
    # Grid Search
    print("Starting Grid Search...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        model_pipeline,
        param_grid,
        cv=cv,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=2
    )
    grid_search.fit(Xtrain, ytrain)

    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.4f}")

    # Logging all parameter combinations
    results = grid_search.cv_results_
    for i in range(len(results['params'])):
        param_set = results['params'][i]
        mean_score = results['mean_test_score'][i]
        std_score = results['std_test_score'][i]
        
        # Logging each combination as a separate MLflow run
        with mlflow.start_run(nested=True):
            mlflow.log_params(param_set)
            mlflow.log_metric("mean_cv_score", mean_score)
            mlflow.log_metric("std_cv_score", std_score)
    
    print(f"✅ Successfully logged all {len(results['params'])} parameter combinations")
    
    # Logging best parameters in main run
    mlflow.log_params(grid_search.best_params_)

    # Storing and evaluating the best model
    best_model = grid_search.best_estimator_

    # Predictions with threshold optimization
    y_proba_test = best_model.predict_proba(Xtest)[:, 1]
    y_pred_test = (y_proba_test >= 0.5).astype(int)  # Default threshold

    # Calculate metrics
    roc_auc = roc_auc_score(ytest, y_proba_test)
    precision, recall, _ = precision_recall_curve(ytest, y_proba_test)
    pr_auc = auc(recall, precision)

    # Classification reports
    test_report = classification_report(ytest, y_pred_test, output_dict=True)

    # Logging metrics
    mlflow.log_metrics({
        "test_roc_auc": roc_auc,
        "test_pr_auc": pr_auc,
        "test_accuracy": test_report['accuracy'],
        "test_precision_1": test_report['1']['precision'],
        "test_recall_1": test_report['1']['recall'],
        "test_f1_score_1": test_report['1']['f1-score'],
        "test_precision_0": test_report['0']['precision'],
        "test_recall_0": test_report['0']['recall'],
        "test_f1_score_0": test_report['0']['f1-score']
    })

    # saving the model locally
    os.makedirs("tourism_project/artifacts", exist_ok=True)
    model_filename = "best_wellness_tourism_model.joblib"
    model_path = f"tourism_project/artifacts/{model_filename}"
    joblib.dump(best_model, model_path)

    # Logging model artifact to MLflow
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved locally at: {model_path}")

    # Uploading mode to hugging face
    print(f"\nUploading model to Hugging Face: {MODEL_REPO_ID}")

    # Checking if model repository exists
    try:
        api.repo_info(repo_id=MODEL_REPO_ID, repo_type="model")
        print(f"Model repository '{MODEL_REPO_ID}' already exists.")
    except RepositoryNotFoundError:
        print(f"Model repository '{MODEL_REPO_ID}' not found. Creating...")
        create_repo(repo_id=MODEL_REPO_ID, repo_type="model", private=False)
        print(f"Model repository '{MODEL_REPO_ID}' created.")

    # Upload model file
    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo=model_filename,
        repo_id=MODEL_REPO_ID,
        repo_type="model"
    )
    print(f"✅ Model uploaded successfully to Hugging Face: {MODEL_REPO_ID}")

    # Summary
    print(f"Best Model ROC AUC: {roc_auc:.4f}")
    print(f"Best Model PR AUC: {pr_auc:.4f}")
    print(f"Test Accuracy: {test_report['accuracy']:.4f}")
    print(f"Precision (Class 1): {test_report['1']['precision']:.4f}")
    print(f"Recall (Class 1): {test_report['1']['recall']:.4f}")
    print(f"F1-Score (Class 1): {test_report['1']['f1-score']:.4f}")
    print(f"Model saved to: {model_path}")
    print(f"Model uploaded to: https://huggingface.co/{MODEL_REPO_ID}")
