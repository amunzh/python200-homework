# --- LLMs as Transform ---
#Q1
# 1. Deterministic code: when we transform dates, we can use a function that provides a stable date format.
# 2. LLM: it requires text classification(understanding) to classify this sentence, which is better done by an LLM
# 3. Deterministic code: a math operation that is easily done by one function; no need for an LLM
# 4. LLM. This is a field extraction, which is better done by an LLM, but if all job titles are stored in one format, Deterministic code could also work
# 5. Deterministic code: a simple count function should be used for this task

#Q2
#This is vague; no certain format was established, and LLM can create different results each time. Because of that, each time in the future when we would want to work with the results, we would have to use the LLM again for text extraction
# Better answer would be:
# system = "Summarize and categorize this product review in a few sentences.
#           Reply with valid JSON only, using the following format:
#            {
#                'sentiment':'positive'/'neutral'/'negative'
#                'category':'product'/'shipping'/'other'
#                'summary': 1-2 sentence summary of the review
#            }"

#Q3
# 1. It will take about 50,000 seconds, or 13.9 hours
# 2. One strategy is to group API calls into batches and send them in parallel, which will significantly reduce the time

# --- Azure OpenAI ---
#Q1
# First is the security reason. Azure OpenAI keeps all the requests in the Azure infrastructure, so data doesn't go to the OpenAI servers, like it doest with calling Open API
# Second is that it is easier that all the data and OpenAI stay under one organization/infrustructure, which is easier to use and in billing

#Q2
# azure_endpoint="https://<resource-name>.openai.azure.com" - This is URL for Azure OpenAI source
# api_key="<azure-api-key>" - Azure OpenAI key to autificate the request
# aapi_version="2024-02-01" - The version that OpenAI API should use
# deployment name(in response) -  Instead of model name uses deployment name held by organization

#Q3
# Instead, it takes model="my-gpt4o-deployment" - which is the deployment name and is set up by the organization's admin.