# Video
# https://drive.google.com/file/d/1kh8wepHwypIluCNak79gEYN_a2Bzwql4/view?usp=sharing

#Part 6
#Classifying weather conditions with the data we had wasn’t a good use of an LLM. Because we had specific numbers for temperature and precipitation, we could’ve used a regular if-else system (deterministic code). 
#The cons of this approach: Much quicker; we do not need to wait for the API calls. Cheaper; we use regular code that could be done anywhere. Reliability: with the same parameters, there will be the same output.
#The pros: Because it is a strict script, in some cases, an LLM can handle a combination of precipitation and temperature better. 

import os
import pandas as pd
from azure.storage.blob import ContainerClient
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv
from openai import OpenAI
import requests
import json

# Part 1
ACCOUNT_URL = "https://anastasiiactd2026sa.blob.core.windows.net"
CONTAINER = "pipeline-data"
credential = DefaultAzureCredential()
container = ContainerClient(
    account_url=ACCOUNT_URL,
    container_name=CONTAINER,
    credential=credential
)
blob_path = f"raw/2026-06-02/weather.json"
raw = container.download_blob(blob_path).readall()
data = json.loads(raw.decode("utf-8"))
hourly = data["hourly"]
records = []
for i in range(len(hourly["time"])):
    record = {
        "time": hourly["time"][i],
        "temperature_2m": hourly["temperature_2m"][i],
        "precipitation": hourly["precipitation"][i],
    }
    records.append(record)

print(f"Loaded {len(records)} hourly records")

#Part 2
SYSTEM_PROMPT = (
    "You are classifying hourly weather conditions for outdoor running. "
    "Given a temperature in Celsius and a precipitation amount in mm, "
    "classify the conditions as exactly one of: good, marginal, or bad. "
    "Reply with that one word only -- no punctuation, no explanation."
)
VALID_LABELS = {"good", "marginal", "bad"}
records_24 = records[:24]

def make_user_message(records_24):
    return (
        f"Temperature: {records_24['temperature_2m']}C, "
        f"Precipitation: {records_24['precipitation']}mm"
    )

load_dotenv()
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
enriched = []
for i, record in enumerate(records_24):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": make_user_message(record)},
        ]
    )
    r_label = response.choices[0].message.content.strip().lower()
    label = r_label if r_label in VALID_LABELS else "unknown"
    enriched_record = {**record, "conditions": label}
    enriched.append(enriched_record)
    if (i + 1) % 6 == 0:
        print(f"  Processed {i + 1} records...")

# Part 3
processed_path = f"processed/2026-06-02/weather_classified.json"
container.upload_blob(processed_path, json.dumps(enriched).encode("utf-8"), overwrite=True)
print(f"Uploaded to {processed_path}")

# Part 4
raw_classified= container.download_blob(processed_path).readall()
data_back = json.loads(raw_classified.decode("utf-8"))
df = pd.DataFrame(data_back)
df["conditions"].value_counts()
print(df.head(5))

# Part 5
with open("outputs/first_10_records.json", "w", encoding="utf-8") as f:
    json.dump(enriched[:10], f)


