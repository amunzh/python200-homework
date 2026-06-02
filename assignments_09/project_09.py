# Video
# https://drive.google.com/file/d/1hDDeDyALfvTVCYC_5PEzLxFYZa8Q9Laa/view?usp=sharing
import io
import pandas as pd
from azure.storage.blob import ContainerClient
from azure.identity import DefaultAzureCredential
import requests
import json
from datetime import date

ACCOUNT_URL = "https://anastasiiactd2026sa.blob.core.windows.net"
CONTAINER = "pipeline-data"

#Part 1
url = (
    "https://api.open-meteo.com/v1/forecast"
    "?latitude=40.7128&longitude=74.0060"
    "&hourly=temperature_2m,precipitation"
    "&forecast_days=7"
)
response = requests.get(url)
response.raise_for_status()
data = response.json()

#Part 2
payload = json.dumps(data).encode("utf-8")

# Part 3
today_date = date.today().isoformat()
blob_path = f"raw/{today_date}/weather.json"

credential = DefaultAzureCredential()
container = ContainerClient(
    account_url=ACCOUNT_URL,
    container_name=CONTAINER,
    credential=credential
)

container.upload_blob(blob_path, payload, overwrite=True)
print(f"Uploaded {len(payload)} bytes to {blob_path}")

# Part 4
def list_container(container_client):
    for blob in container_client.list_blobs():
        print(f'{blob.name} {blob.size} bytes')
list_container(container)

# Part 5
raw = container.download_blob(blob_path).readall()
data_back = json.loads(raw.decode("utf-8"))
df = pd.DataFrame(data_back["hourly"])
print(df.head(5))

with open("outputs/weather_raw.json", "wb") as f:
    f.write(raw)
