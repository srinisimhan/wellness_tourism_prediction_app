from huggingface_hub import HfApi
import os

# HF Authentication
HF_TOKEN = os.getenv("HF_TOKEN")
# Strip any potential whitespace from the token
if HF_TOKEN:
    HF_TOKEN = HF_TOKEN.strip()
api = HfApi(token=HF_TOKEN)

api.upload_folder(
    folder_path="tourism_project/deployment", 
    repo_id="simnid/Wellness-Tourism-Prediction",     
    repo_type="space",                         
    path_in_repo="",                           
)
