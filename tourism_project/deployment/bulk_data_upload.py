from huggingface_hub import HfApi
import os
import pandas as pd

# creating bulk test data and saving locally 
# Define sample bulk data
bulk_data = [
    [35,"Self Enquiry",2,15.0,"Salaried","Male",2,3,"Deluxe",4,"Married",2,1,3.0,1,0,"Manager",15000,45.0],
    [50,"Company Invited",3,30.0,"Large Business","Female",1,1,"Standard",5,"Single",5,1,4.5,0,1,"VP",35000,135.0],
    [28,"Self Enquiry",1,10.0,"Small Business","Male",3,0,"Basic",3,"Unmarried",1,0,2.0,1,2,"Executive",12000,20.0]
]

columns = [
    'Age','TypeofContact','CityTier','DurationOfPitch','Occupation','Gender',
    'NumberOfPersonVisiting','NumberOfFollowups','ProductPitched','PreferredPropertyStar',
    'MaritalStatus','NumberOfTrips','Passport','PitchSatisfactionScore','OwnCar',
    'NumberOfChildrenVisiting','Designation','MonthlyIncome','PitchEfficiency'
]

df_bulk = pd.DataFrame(bulk_data, columns=columns)

# Save locally
local_path = "tourism_project/data/bulk_test_sample.csv"
df_bulk.to_csv(local_path, index=False)
print(f"Bulk CSV saved locally at {local_path}")

# Get access token from local
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN:
    HF_TOKEN = HF_TOKEN.strip()
else:
    raise EnvironmentError("HF_TOKEN not set!")

DATA_REPO_ID = "simnid/wellness-tourism-dataset"
BULK_CSV_PATH = "tourism_project/data/bulk_test_sample.csv"
BULK_FILENAME = "bulk_test_sample.csv"

api = HfApi(token=HF_TOKEN)

# Upload CSV
api.upload_file(
    path_or_fileobj=BULK_CSV_PATH,
    path_in_repo=BULK_FILENAME,
    repo_id=DATA_REPO_ID,
    repo_type="dataset",
    token=HF_TOKEN
)

print(f"Bulk CSV uploaded to Hugging Face dataset repo: {DATA_REPO_ID}/{BULK_FILENAME}")
