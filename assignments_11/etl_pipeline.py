import os
from dotenv import load_dotenv
from azure.storage.blob import ContainerClient
from azure.identity import DefaultAzureCredential
import requests
from openai import OpenAI
from prefect import flow, task
from prefect.logging import get_run_logger
from datetime import date
import json

load_dotenv()
ACCOUNT_URL = "https://anastasiiactd2026sa.blob.core.windows.net"
CONTAINER = "pipeline-data"

@task(retries=2, retry_delay_seconds=10)
def extract(latitude: float, longitude: float) -> dict:
    url = (
    f"https://api.open-meteo.com/v1/forecast"
    f"?latitude={latitude}&longitude={longitude}"
    f"&hourly=temperature_2m,precipitation"
    f"&forecast_days=7") # New York

    response = requests.get(url)
    response.raise_for_status()
    print(f"Extracted forecast data for New York({latitude}, {longitude})")
    return response.json()


@task
def transform(data: dict) -> list:
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

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

    records = records[:24] # First day only
    SYSTEM_PROMPT = ("You are classifying hourly weather conditions for outdoor running."
        "Given a temperature in Celsius and a precipitation amount in mm,"
        "classify the conditions as exactly one of: good, marginal, or bad."
        "Reply with that one word only -- no punctuation, no explanation.")
    VALID_LABELS = {"good", "marginal", "bad"}
    
    enriched = []
    for i, record in enumerate(records):
        user_m = (
            f"Temperature: {record['temperature_2m']}C, "
            f"Precipitation: {record['precipitation']}mm"
        )

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_m},
            ]
        )

        r_label = response.choices[0].message.content.strip().lower()
        label = r_label if r_label in VALID_LABELS else "unknown"

        enriched_record = {**record, "conditions": label}
        enriched.append(enriched_record)

        if (i + 1) % 6 == 0:
            print(f"  Processed {i + 1} records...")

    print(f"Transform finished: {len(enriched)} records")
    return enriched


@task
def load(records: list,blob_path: str) -> None:
    logger = get_run_logger()
    credential = DefaultAzureCredential()
    container = ContainerClient(
        account_url=ACCOUNT_URL,
        container_name=CONTAINER,
        credential=credential)

    container.upload_blob(blob_path, json.dumps(records).encode("utf-8"), overwrite=True)
    logger.info(f"Uploaded {len(json.dumps(records).encode("utf-8"))} bytes to {blob_path}")

@flow(log_prints=True)
def etl_pipeline():
    lat = 40.7128
    long = 74.0060
    today = date.today().isoformat()
    blob_path = f'final/{today}/weather_etl.json'
    
    data = extract(lat,long)
    encr = transform(data)
    load(encr,blob_path)
    print(f'Completed successfully! Results: {blob_path}')

if __name__ == "__main__":
    etl_pipeline()

