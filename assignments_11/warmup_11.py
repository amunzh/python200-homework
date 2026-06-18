# --- Prefect Orchestration ---
# Q1
# @task is a single unit of work; there can be multiple of them, and it can load data, transform, call an API, or write a file. While @flow calls all the @task in order and with specific inputs.
# No, I would not decorate this function with @task because it is a simple function that requires only completing simple math steps, so using Perfect here is completely unnecessary.

#Q2
# @task(name = "call_api", retries=3, retry_delay_seconds=30)

#Q3
# I would look at what went wrong in the transform stage because it indicated that it failed, which resulted in the load stage not being able to proceed further. 
# I would open the transform stage in Prefect UI, check how many retries there were, and the detailed run to see where exactly the problem was. I would expect that there could be a problem with an API key or an OpenAI problem. But there could be other programming bugs.

# --- Production Patterns ---
#Q1
# raise_for_status() shows any exceptions if the request failed. It is better than just using if response.status_code != 200: print("error") because it actually stops the pipeline, not just prints the message, which prevents storing/working with bad data at the start. 
# In case the API returns a 500 error, if response.status_code != 200: print("error") will just continue with the program, which can cause trouble. On the other hand, raise_for_status will stop the pipeline and indicate that the API returned an error. 

#Q2
# Without overwrite=True, rerunning the program would fail because the file already exists. With overwrite=True, the program will rerun without problem because each time it replaces the old file with a new one, allowing the program to run successfully multiple times

#Q3
# @task
# def load(records: list, blob_path: str) -> None:
#    logger = get_run_logger()
#    logger.info(f"Loaded {len(records)} records to {blob_path}")

