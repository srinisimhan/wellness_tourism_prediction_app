# for data manipulation
import pandas as pd
# for creating a folder
import os
# for data preprocessing and pipeline creation
import sklearn
from sklearn.model_selection import train_test_split
# for converting text data in to numerical representation
from sklearn.preprocessing import LabelEncoder
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

# HF Authentication
HF_TOKEN = os.getenv("HF_TOKEN")
# Strip any potential whitespace from the token
if HF_TOKEN:
    HF_TOKEN = HF_TOKEN.strip()
api = HfApi(token=HF_TOKEN)

# Dataset repo
DATA_REPO_ID = "simnid/wellness-tourism-dataset"
DATA_FILE_PATH = "hf://datasets/simnid/wellness-tourism-dataset/tourism.csv"

# Define output directory
output_dir = "tourism_project/data"
os.makedirs(output_dir, exist_ok=True)

# Load dataset
df = pd.read_csv(DATA_FILE_PATH)
print("Dataset loaded successfully from Hugging Face.")

# Drop columns
cols_to_drop = ["CustomerID", "Unnamed: 0"]
df.drop(columns=cols_to_drop, errors='ignore', inplace=True)

# Fix typo
if "Gender" in df.columns:
    df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})

# Feature engineering
df["PitchEfficiency"] = df["PitchSatisfactionScore"] * df["DurationOfPitch"]

# Encode categorical variables
cat_cols = df.select_dtypes(include="object").columns
label_encoders = {}

for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# identifying the target variable
target = "ProdTaken"
X = df.drop(columns=[target])
y = df[target]

# Split into train and test datasets.
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Save locally
Xtrain.to_csv(os.path.join(output_dir, "Xtrain.csv"), index=False)
Xtest.to_csv(os.path.join(output_dir, "Xtest.csv"), index=False)
ytrain.to_csv(os.path.join(output_dir, "ytrain.csv"), index=False)
ytest.to_csv(os.path.join(output_dir, "ytest.csv"), index=False)

print("Train/test datasets created & saved locally.")

# Upload back to Hugging Face
files = ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]

for file_name in files:
    file_path = os.path.join(output_dir, file_name)
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_name,
        repo_id=DATA_REPO_ID,
        repo_type="dataset"
    )
    print(f"Uploaded {file_name} to {DATA_REPO_ID}")

print("✔ All processed files uploaded successfully.")
