# Cloud Concepts
1. Cloud computing lets you rent computing infrastructure instead of buying everything. You only pay for what you use on demand and it is not restricted by the hardware space you own

2. Vertical scaling upgrades a computer's hardware(RAM, CPU, GPU). For example, it could be useful when the software is too slow, and we want it to work faster. Horizontal scaling adds more machines to the system, which is useful if there is an increase in traffic. 
For the first scenario, horizontal scaling would be better, as it will distribute the traffic and prevent it from crashing 
For the second scenario, vertical scaling applies because the scientist needs more powerful hardware. 
For the 3rd scenario, horizontal scaling would apply because distributing across multiple machines would speed up the process

3. Gmail - SaaS because it's an application that we just use without the need to run and maintain it.
Azure Virtual Machines - IaaS, it provides a raw computing resource that we have to manage(software, updates, and so on)
Azure App Service - PaaS, User deploying code while the app manages the infrastructure
AWS S3 (Simple Storage Service) - PaaS. It is a storage service that makes it harder to separate. The app can manage the storage and how long the info will last, but it requires a built-in application
GitHub Codespaces - PaaS gives you a cloud environment where you can manage your code
Snowflake - SaaS, gives a fully managed cloud data platform

IaaS is when you rent computing infrastructure and set it up for yourself and manage everything. For example, in Google Compute Engine, I would have to manage everything in my space, including the operating system, software, and security.
PaaS lets me bring my own code and data, and manage my application while the provider handles servers. For example, GitHub Codespaces, where I upload my own code while Git manages the working space itself.
SaaS model with an already complete working application where I, as a user, only use it. For example, I'm only using Gmail as an app while developers manage everything.

4. Managed data platforms is a layer that helps set up and manage the cloud service for you, while if using a cloud provider, you have to set up everything yourself.
With managed data platforms, you save time and costs, and it is much simpler, but you lose flexibility and don't have a lot of freedom.

5. If dataset doesn't require data storage and if computations can simply be handalled on one machine.

# Azure Basics
1. Azure subscription is billing account(CTD) and within that subscription there's resource group(mine) that orginizes Azure resources.

2. Ephemeral means every time I close the shell, everything will be deleted. In the setup, we connected Cloud Shell to a file share so that everything would be saved from session to session

3. SSH private key stays in my computer and is never shared, while the public key gets uploaded to the system. When connecting to a remote system, SSH verifies keys, and the system knows who it is without any credentials.

4. {
  "environmentName": "AzureCloud",
  "homeTenantId": "0f040ddd-301f-4665-8677-7b21f129d605",
  "id": "4e07c58c-751e-4765-b40c-632b9ee6fe6e",
  "isDefault": true,
  "managedByTenants": [],
  "name": "CTD Nonprofit Sponsorship",
  "state": "Enabled",
  "tenantId": "0f040ddd-301f-4665-8677-7b21f129d605",
  "user": {
    "cloudShellID": true,
    "name": "live.com#unzhakova.a.m@gmail.com",
    "type": "user"
  }
}
--output table command produces a readable table instead of a json 