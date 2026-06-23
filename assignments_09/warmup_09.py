# --- Azure Authentication ---
# Q1
# DefaultAzureCredential tries a sequence of authentication methods and stops at the right one. First, we have to run az login so it can pick up the session automatically(picking up the az login session is part of the DefaultAzureCredential process)

#Q2 
# Because az login relies on a human developing the session and cannot be used for a pipeline. Instead, it uses managed identity; the DefaultAzureCredential can see it and use it without human input because it tries different authentication methods, and after finding the right one, the code will run the same no matter what authentication method it uses.

#Q3
# Most likely, the azur session expired, and we need to run az login again. Another possibility is that I'm running on a different subscription, and in this case, we need to check the subscription using
# for sub in client.subscriptions.list():
#    print(sub.display_name)


# --- Blob Storage ---
#Q1
# First, a storage account is a root(top-level) resource, like a hard drive; it gives a security boundary and a unique URL
# A container contains a set of blobs, like a folder; it groups different files, and there can be many containers under one storage account. Usually, one container is one project or pipeline
# Blob is an individual file itself(inside folder-container) it can be pdf,json, mp4 and so on.

# Q2 
# 1. Blob storage because we need to STORE raw responses; we only save files and don't need to query that data.
# 2. Relational database because the analytical team needs to query data constantly, and Blob storage works with files; it doesn't filter rows/columns
# 3. Blob storage because we only need to SAVE NumPy arrays(files) between pipeline runs.

#Q3
def list_container(container_client):
    for blob in container_client.list_blobs():
        print(blob.name, blob.size)

#Q4
def upload_text(container_client, blob_name, text):
    container_client.upload_blob(blob_name, text.encode("utf-8"), overwrite=True)