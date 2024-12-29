# Project ProjectName

$$\color{red}{IMPORTANT}$$
<span style="color:red"> This table is necessary to update </span>

| Version | name | Release Date | Description |
| ------- |---------| ------------ | ----------- |
| 1.0     | Luis Galo Corzo |February 13, 2023 | Initial release |
<!-- PULL_REQUESTS_TABLE -->
<!-- cspell:ignore Databricks  -->
<!-- cspell:disable -->

## Contents

- [Project ProjectName](#project-projectname)
  - [Contents](#contents)
  - [Introduction](#introduction)
  - [Scope](#scope)
  - [Objectives](#objectives)
  - [rag for codeBert](#rag-for-codebert)
  - [codebert finetune](#codebert-finetune)
  - [Requirements](#requirements)
  - [Structure of the repo](#structure-of-the-repo)
  - [testing](#testing)
  - [Keywords](#keywords)
  - [Frontend](#frontend)
  - [references](#references)
  - [end to end  ml project](#end-to-end--ml-project)
  - [diagrams](#diagrams)
  - [install nginx and oauth2.0](#install-nginx-and-oauth20)
  - [connect to Oauth2.0 to github](#connect-to-oauth20-to-github)
  - [DVC installation:](#dvc-installation)

<!-- cspell:enable -->
## Introduction

TThe goal of this project is to create a Template  code example and folder structure  that make easy the initial process of starting the development of a new ML service. This template generates an example with test, notebooks and pipelines to ensure the code quality

## Scope

- The scope of the project includes the creation of all the stages defined in the MLOPS process the get a ML model scalable and ready to be deployed in production

- The docker instructions to deploy a local example for testing proposes

- the code to deploy the model as a service in the cloud wit Mlflow

## Objectives

- The main objective of the project is to create a Databricks infrastructure that allows for the efficient transfer of data to an MLOps system in Azure.
- The project will also aim to ensure that data is properly formatted, cleaned, and transformed into a format that is usable by machine learning algorithms.
- The project will also aim to monitor and update the data pipeline to ensure that the data is up-to-date and accurate.
  
-----------------------------------------------------
TBD

## rag for codeBert

[rag for codebert](https://medium.com/@ashwin_rachha/querying-a-code-database-to-find-similar-coding-problems-using-langchain-814730da6e6d#id_token=eyJhbGciOiJSUzI1NiIsImtpZCI6IjkxNDEzY2Y0ZmEwY2I5MmEzYzNmNWEwNTQ1MDkxMzJjNDc2NjA5MzciLCJ0eXAiOiJKV1QifQ.eyJpc3MiOiJodHRwczovL2FjY291bnRzLmdvb2dsZS5jb20iLCJhenAiOiIyMTYyOTYwMzU4MzQtazFrNnFlMDYwczJ0cDJhMmphbTRsamRjbXMwMHN0dGcuYXBwcy5nb29nbGV1c2VyY29udGVudC5jb20iLCJhdWQiOiIyMTYyOTYwMzU4MzQtazFrNnFlMDYwczJ0cDJhMmphbTRsamRjbXMwMHN0dGcuYXBwcy5nb29nbGV1c2VyY29udGVudC5jb20iLCJzdWIiOiIxMTc1MDkxNTA1ODY5ODk4NTI5MDkiLCJlbWFpbCI6ImxnY29yem9AZ21haWwuY29tIiwiZW1haWxfdmVyaWZpZWQiOnRydWUsIm5iZiI6MTcwNDkyMTE0OSwibmFtZSI6Ikx1aXMgR2FsbyIsInBpY3R1cmUiOiJodHRwczovL2xoMy5nb29nbGV1c2VyY29udGVudC5jb20vYS9BQ2c4b2NMQmw1N3ZiVFZSelllV3M1ZXlVdFBhbVB5OXA2dXNKdnBSNURqQzlYMmc9czk2LWMiLCJnaXZlbl9uYW1lIjoiTHVpcyIsImZhbWlseV9uYW1lIjoiR2FsbyIsImxvY2FsZSI6ImVzIiwiaWF0IjoxNzA0OTIxNDQ5LCJleHAiOjE3MDQ5MjUwNDksImp0aSI6IjQ1MGU4NWVjZGFkNjI3NDA0N2ZiNzg0MjUzZWY1Y2FhMzNiZjg4OTcifQ.QwGpfy4gNBmU54TjU93XGjvwidLSwIHNFtDhOhYPg8f5NdU32bZUOgeWtaNgXKKRQsHuCfr5JpROjp60rOdO1vhPny4PqKciFPNNCPgoCYK0KuSd1LMGT7CwGrHHH515kuCtxY72PlvtthcQz5CDaHuRf50tUZv5CTJB8ueCTNDNWsAOSWul-nXEWqYZT-WpBih7ZEIuYhU1HXa7TN6CAdeLe92eDi1giJCdabPrkp0HXgv0DmGywfVwJ77PEVp3rgdXDA6L0IC5Mj1_7vTb_SeAjPMKoD1_vfOtuDcjeuIUXp3K-ZKAdZPFH63eqqPaOnKtkNioXiy3sVxptw9byQ)

Here is a possible brief of the page for your readme:

This page is about how to use large language models (LLMs) such as CodeBERT and GPT-3 for code intelligence tasks, such as code similarity, code search, and code generation¹[1]. It also introduces LangChain, a Python library that provides out-of-the-box support to build NLP applications using LLMs²[2]. You can connect to various data and computation sources, and build applications that perform NLP tasks on domain-specific data sources, private repositories, and more³[3].

The page contains the following sections:

- **Introduction**: This section gives an overview of the motivation and challenges of using LLMs for code intelligence, and the main features and use cases of LangChain.
- **CodeBERT**: This section explains what CodeBERT is, how it works, and how to use it for code similarity tasks. It also shows how to use the langchain library to load and split code snippets from a CSV file, generate embeddings for code using CodeBERT, and create a FAISS index for similarity search⁴[4].
- **GPT-3**: This section explains what GPT-3 is, how it works, and how to use it for code generation tasks. It also shows how to use the langchain library to create prompts and chains for different code generation scenarios, such as generating code from natural language, generating code from pseudocode, and generating code from test cases.
- **Conclusion**: This section summarizes the main points of the page and provides some links to the code and resources used in the examples.

A RAG (retrieval-augmented generation) is a method of using large language models (LLMs) to generate text based on external data sources, such as documents or code snippets. To make a RAG with CNC codes using CodeBERT LLM, you would need to follow these steps:

- Prepare a data warehouse that contains CNC codes and other relevant information for your task. You can use any format that is compatible with CodeBERT, such as Python, Java, C#, etc.
- Use CodeBERT to embed the CNC codes and other data chunks in your data warehouse into vector representations. You can use the `encode` method of the CodeBERT model to do this.
- Use a vector retrieval system, such as FAISS, to index and search the embedded data chunks based on a given query. You can use the `search` method of the FAISS index to do this.
- Use CodeBERT to generate a response based on the query and the retrieved data chunks. You can use the `generate` method of the CodeBERT model to do this.

Here is a possible code example of how to make a RAG with CNC codes using CodeBERT LLM:

```python
# Import the required libraries
import torch
from transformers import AutoTokenizer, AutoModel
import faiss

# Load the CodeBERT model and tokenizer
model = AutoModel.from_pretrained("microsoft/codebert-base")
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")

# Define a function to encode data chunks into vectors
def encode(data):
  inputs = tokenizer(data, return_tensors="pt", padding=True)
  outputs = model(**inputs)
  embeddings = outputs.last_hidden_state[:,0,:]
  return embeddings.detach().numpy()

# Load the data warehouse from a CSV file
import pandas as pd
data = pd.read_csv("cnc_data.csv")

# Encode the data chunks using CodeBERT
data["embeddings"] = data["code"].apply(encode)

# Flatten the embeddings into a 2D array
embeddings = np.vstack(data["embeddings"].values)

# Create a FAISS index for vector retrieval
d = embeddings.shape[1]
index = faiss.IndexFlatL2(d)
index.add(embeddings)

# Define a function to generate a response based on a query and retrieved data chunks
def generate(query, k=5):
  # Encode the query into a vector
  query_embedding = encode(query)
  # Retrieve the top k most similar data chunks
  D, I = index.search(query_embedding, k)
  # Concatenate the query and the retrieved data chunks
  input_ids = tokenizer(query, return_tensors="pt").input_ids
  for i in I[0]:
    input_ids = torch.cat([input_ids, tokenizer(data["code"][i], return_tensors="pt").input_ids], dim=-1)
  # Generate a response using CodeBERT
  output_ids = model.generate(input_ids, max_length=256)
  output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
  return output

# Test the RAG with a sample query
query = "How to make a circle with CNC code?"
response = generate(query)
print(response)
```

Output:

```
How to make a circle with CNC code?

One way to make a circle with CNC code is to use the G02 and G03 commands, which are used for clockwise and counterclockwise circular interpolation, respectively. These commands require the following parameters:

- X and Y: the coordinates of the end point of the arc
- I and J: the offsets of the center of the arc from the current position
- F: the feed rate

For example, the following CNC code will make a circle with a radius of 10 units and a center at (10, 10):

G00 X0 Y0 ; move to the starting point
G02 X20 Y0 I10 J0 F100 ; make a clockwise arc to the right
G02 X20 Y20 I0 J10 F100 ; make a clockwise arc to the top
G02 X0 Y20 I-10 J0 F100 ; make a clockwise arc to the left
G02 X0 Y0 I0 J-10 F100 ; make a clockwise arc to the bottom
```

## codebert finetune

<https://colab.research.google.com/github/Ankur3107/colab_notebooks/blob/master/classification/BERT_Fine_Tuning_Sentence_Classification_v2.ipynb>

To fine-tune CodeBERT for this task, we need to follow these steps:

- Load the pre-trained CodeBERT model from [HuggingFace](^3^) or [GitHub].
- Add a regression head on top of the model. This can be a linear layer that takes the pooled output of the last hidden layer of CodeBERT and produces a scalar value representing the cosine distance between two input documents.
- Define a loss function and an optimizer. We can use mean squared error loss and Adam optimizer for this task.
- Prepare a dataset of document pairs with their corresponding cosine distances. We can use existing datasets such as [STS-B] or [SICK], or create our own dataset by computing the cosine distances between documents using [scikit-learn](^1^) or other libraries.
- Split the dataset into training, validation, and test sets. Shuffle and batch the data for each set.
- Train the model on the training set for a few epochs (2-4 recommended). Evaluate the model on the validation set after each epoch and save the best checkpoint based on the validation loss.
- Test the model on the test set using the best checkpoint. Report the test loss and other metrics such as Pearson correlation, Spearman correlation, and mean absolute error.

Here is a code snippet that shows how to fine-tune CodeBERT for this task using PyTorch:

```python
# Import libraries
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd

# Load pre-trained CodeBERT model and tokenizer
model = AutoModel.from_pretrained("microsoft/codebert-base")
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")

# Add a regression head
model.classifier = nn.Linear(model.config.hidden_size, 1)

# Define device, loss function, optimizer, and scheduler
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
criterion = nn.MSELoss()
optimizer = AdamW(model.parameters(), lr=2e-5)
num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

# Define a custom dataset class
class DocumentPairDataset(Dataset):
    def __init__(self, data_file, tokenizer, max_length):
        self.data = pd.read_csv(data_file)
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        row = self.data.iloc[index]
        doc1 = row["doc1"]
        doc2 = row["doc2"]
        label = row["label"]
        inputs = self.tokenizer.encode_plus(doc1, doc2, add_special_tokens=True, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")
        input_ids = inputs["input_ids"].squeeze()
        attention_mask = inputs["attention_mask"].squeeze()
        token_type_ids = inputs["token_type_ids"].squeeze()
        label = torch.tensor(label).float()
        return input_ids, attention_mask, token_type_ids, label

# Load data and create data loaders
train_dataset = DocumentPairDataset("train.csv", tokenizer, max_length=128)
val_dataset = DocumentPairDataset("val.csv", tokenizer, max_length=128)
test_dataset = DocumentPairDataset("test.csv", tokenizer, max_length=128)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32)
test_dataloader = DataLoader(test_dataset, batch_size=32)

# Train and evaluate the model
best_val_loss = float("inf")
for epoch in range(num_epochs):
    # Train
    model.train()
    train_loss = 0.0
    for batch in train_dataloader:
        input_ids, attention_mask, token_type_ids, labels = [b.to(device) for b in batch]
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        logits = outputs.logits.squeeze()
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        train_loss += loss.item()
    train_loss /= len(train_dataloader)
    print(f"Epoch {epoch+1}, Train loss: {train_loss:.4f}")
    
    # Evaluate
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_dataloader:
            input_ids, attention_mask, token_type_ids, labels = [b.to(device) for b in batch]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            logits = outputs.logits.squeeze()
            loss = criterion(logits, labels)
            val_loss += loss.item()
        val_loss /= len(val_dataloader)
        print(f"Epoch {epoch+1}, Val loss: {val_loss:.4f}")
        # Save the best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pt")

# Test the model using the best checkpoint
model.load_state_dict(torch.load("best_model.pt"))
model.eval()
test_loss = 0.0
predictions = []
labels = []
with torch.no_grad():
    for batch in test_dataloader:
        input_ids, attention_mask, token_type_ids, labels = [b.to(device) for b in batch]
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        logits = outputs.logits.squeeze()
        loss = criterion(logits, labels)
        test_loss += loss.item()
        predictions.extend(logits.cpu().numpy())
        labels.extend(labels.cpu().numpy())
    test_loss /= len(test_dataloader)
    print(f"Test loss: {test_loss:.4f}")
    # Compute other metrics
    predictions = np.array(predictions)
    labels = np.array(labels)
    pearson_corr = np.corrcoef(predictions, labels)[0, 1]
    spearman_corr = scipy.stats.spearmanr(predictions, labels).correlation
    mean_abs_error = np.mean(np.abs(predictions - labels))
    print(f"Pearson correlation: {pearson_corr:.4f}")
    print(f"Spearman correlation: {spearman_corr:.4f}")
    print(f"Mean absolute error: {mean_abs_error:.4f}")
```

Source: Conversation with Bing, 10/01/2024
(1) Building RAG-based LLM Applications for Production. <https://www.anyscale.com/blog/a-comprehensive-guide-for-building-rag-based-llm-applications-part-1>.
(2) GitHub - pchunduri6/rag-demystified: An LLM-powered advanced RAG .... <https://github.com/pchunduri6/rag-demystified>.
(3) No-code retrieval augmented generation (RAG) with LlamaIndex and .... <https://bdtechtalks.com/2023/11/22/rag-chatgpt-llamaindex/>.
(4) GitHub - run-llama/rags: Build ChatGPT over your data, all with natural .... <https://github.com/run-llama/rags>.

## Requirements

- A Databricks infrastructure that can collect, process, and store data from various sources.
- An MLOps system in Azure that can receive and process data from the Databricks infrastructure.
- A data pipeline that automates the collection and preprocessing of data.
- A system for monitoring and updating the data pipeline to ensure that the data is up-to-date and accurate.

## Structure of the repo

The windows command to extract the folder structure is:

 ``` cmd
 tree /a /f > output.txt.
```

The initial content is:

 ``` txt
|   .env (file with the local configuration of the environment variables)
|   .gitattributes (file with git configuration)
|   .gitignore (files to be ignored)
|   README.md (README file with the general information of the project)
|   setup.cfg (setup file for  flake and other tools)
+---Code (folder to put the code to be control by the CI pipelines
|   \---Utils (example folder)
|           __init__.py
+---Data (folder to put the local data of the project)
|       .gitkeep
+---Notebooks (folder to place the experimenting  notebooks)
|   \---DataIngest (example folder)
|           .gitkeep
+---Pipelines (folder for the pipeline YAMLs)
|       .gitkeep
+---Settings (folder for the environments YML files)
|   \---Code
|           mlops_data_ingest_env.yaml (initial env)
\---Tests (folder for the test)
    |   __init__.py
    \---Code
            __init__.py
             \---Utils (example folder for the Utils code folder)
                         __init__.py
```

## testing

to check the coverage of the code install coverage gutter plugin and run:

``` bash
pytest --cov=Code  --cov-report=xml:cov.xml
```

delete pytest cache

``` powershell
Get-ChildItem -Path . -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force
```

```cmd
rmdir /s /q .pytest_cache

```

## Keywords

- Databricks Infrastructure
- MLOps System
- Azure
- Data Sources
- Data Processing
- Data Storage
- Data Pipeline
- Data Monitoring

## Frontend

pyhton frontend with dash ( POC)

professional FE:
angular

- Angular 15 o superior.
- Material Design mediante Angular Material
- Syncfusion Angular UI: libreria de componentes
- Jest: framework de testing

path to the Frontend <http://localhost:9000/frontend-service-dev/>

## references

NLP
<https://medium.com/analytics-vidhya/best-nlp-algorithms-to-get-document-similarity-a5559244b23b>
<https://github.com/jairNeto/warren_buffet_letters/tree/main>

huggingface repo:
<https://github.com/huggingface/transformers/tree/a5cc30d72ae2dc19af534e4b35c986cc28db1275>

<https://huggingface.co/docs/transformers/task_summary>
 for visio and pipelines
<https://theaisummer.com/hugging-face-vit/>

ProjectName MachineConfiguratorFinder
project_name  example machine_configuration_finder

<https://github.com/microsoft/CodeBERT>

[llama-2-open-foundation-and-fine-tuned-chat-models](https://ai.meta.com/research/publications/llama-2-open-foundation-and-fine-tuned-chat-models/)

Codebert
<https://github.com/microsoft/CodeBERT>
<https://huggingface.co/microsoft/codebert-base>

## end to end  ml project

<https://github.com/shanakaChathu/churn_model>
<https://medium.com/@shanakachathuranga/end-to-end-machine-learning-pipeline-with-mlops-tools-mlflow-dvc-flask-heroku-evidentlyai-github-c38b5233778c>

.

Source: Conversation with Bing, 12/08/2023
(1) How to Calculate Cosine Similarity in Python? - GeeksforGeeks. <https://www.geeksforgeeks.org/how-to-calculate-cosine-similarity-in-python/>.
(2) sklearn.metrics.pairwise.cosine_similarity — scikit-learn 1.3.0 .... <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html>.
(3) Cosine Similarity - Understanding the math and how it works? (with python). <https://www.machinelearningplus.com/nlp/cosine-similarity/>.

## diagrams

::: mermaid
classDiagram
  class ProjectNameModel {
    -List[Any] knowledge_tokenized
    -List[Any] knowledge_cnc_name
    -Any pretrained_model
    -Any pretrained_tokenizer
    -pd.DataFrame cnc_df
    +**init**(cnc_path) None
    +load_pretrained_llm(llm_type) None
    +fit(x_in, y_in) ProjectNameModel
    +predict_probea(x_in, num_results) pd.DataFrame
    +score(x_test, y_test) float
    +predict(x_in) pd.Series
  }

:::

--------------------------------------------

## install nginx and oauth2.0

<https://github.com/oktadev/okta-oauth2-proxy-example>

<https://developer.okta.com/blog/2022/07/14/add-auth-to-any-app-with-oauth2-proxy>

<https://medium.com/devops-dudes/using-oauth2-proxy-with-nginx-subdomains-e453617713a>

First, you need to create a new service for nginx in your docker compose file. You can use the official nginx image from Docker Hub¹ or build your own image with a custom configuration. For example, you can use the following service definition:

```yaml
  nginx:
    image: nginx:latest
    ports:
      - 80:80
      - 443:443
    depends_on:
      - frontend
    networks:
      servnet:
        ipv4_address: 172.30.1.9
```

This will create a nginx container that listens on port 80 and 443, and depends on the frontend service. You also need to assign a static IP address to the nginx container in the same network as the other services.

Next, you need to add oauth2-proxy² as a sidecar container to the nginx service. Oauth2-proxy is a reverse proxy that provides authentication and authorization for your web applications using OAuth 2.0 providers. You can use the official oauth2-proxy image from Docker Hub³ or build your own image with a custom configuration. For example, you can use the following service definition:

```yaml
  oauth2-proxy:
    image: quay.io/oauth2-proxy/oauth2-proxy:v7.2.0
    environment:
      OAUTH2_PROXY_CLIENT_ID: <your-client-id>
      OAUTH2_PROXY_CLIENT_SECRET: <your-client-secret>
      OAUTH2_PROXY_COOKIE_SECRET: <your-cookie-secret>
      OAUTH2_PROXY_PROVIDER: <your-provider>
      OAUTH2_PROXY_UPSTREAM: http://172.30.1.9
    ports:
      - 4180:4180
    networks:
      servnet:
        ipv4_address: 172.30.1.10
```

This will create a oauth2-proxy container that listens on port 4180, and proxies requests to the nginx container at 172.30.1.9. You need to replace the environment variables with your own values, depending on your OAuth 2.0 provider. You also need to assign a static IP address to the oauth2-proxy container in the same network as the other services.

Finally, you need to configure the nginx service to use the oauth2-proxy service as a proxy for the frontend application. You can do this by creating a custom nginx.conf file and mounting it as a volume to the nginx container. For example, you can use the following nginx.conf file:

```nginx
events {}
http {
  server {
    listen 80;
    listen 443 ssl;
    server_name _;

    ssl_certificate /etc/nginx/certs/cert.pem;
    ssl_certificate_key /etc/nginx/certs/key.pem;

    location / {
      proxy_pass http://172.30.1.10:4180;
      proxy_set_header Host $host;
      proxy_set_header X-Real-IP $remote_addr;
      proxy_set_header X-Scheme $scheme;
      proxy_set_header X-Auth-Request-Redirect $request_uri;
    }

    location = /oauth2/auth {
      proxy_pass http://172.30.1.10:4180;
      proxy_set_header Host $host;
      proxy_set_header X-Real-IP $remote_addr;
      proxy_set_header X-Scheme $scheme;
      # nginx auth_request includes headers but not body
      proxy_set_header Content-Length "";
      proxy_pass_request_body off;
    }

    location = /oauth2/callback {
      # This is the callback URL registered with the OAuth provider
      # Example: http://auth.example.com/oauth2/callback
      proxy_pass http://172.30.1.10:4180;
      proxy_set_header Host $host;
      proxy_set_header X-Real-IP $remote_addr;
      proxy_set_header X-Scheme $scheme;
    }
  }
}
```

This will configure nginx to use oauth2-proxy as an authentication layer for the frontend application, and redirect users to the OAuth 2.0 provider for login. You need to replace the ssl_certificate and ssl_certificate_key with your own values, and make sure they are available in the /etc/nginx/certs directory. You can mount this directory as a volume to the nginx container. For example, you can use the following volume definition:

```yaml
  nginx:
    image: nginx:latest
    ports:
      - 80:80
      - 443:443
    depends_on:
      - frontend
    networks:
      servnet:
        ipv4_address: 172.30.1.9
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./certs:/etc/nginx/certs
```

This will mount the nginx.conf file and the certs directory from your host machine to the nginx container.

## connect to Oauth2.0 to github

To connect oauth2.0 to GitHub, you need to create an OAuth app on GitHub and configure your oauth2-proxy service to use GitHub as the provider. Here are the steps you can follow:

- Sign in to GitHub and create an OAuth app¹. You can use any name and homepage URL for your app, but you need to set the authorization callback URL to `http://172.30.1.10:4180/oauth2/callback`, which is the URL of your oauth2-proxy service.
- After creating the app, you will get a client ID and a client secret from GitHub. You need to use these values to set the environment variables `OAUTH2_PROXY_CLIENT_ID` and `OAUTH2_PROXY_CLIENT_SECRET` in your oauth2-proxy service definition in your docker compose file.
- You also need to set the environment variable `OAUTH2_PROXY_PROVIDER` to `github` in your oauth2-proxy service definition. This will tell oauth2-proxy to use GitHub as the OAuth 2.0 provider.
- You need to generate a random string and use it to set the environment variable `OAUTH2_PROXY_COOKIE_SECRET` in your oauth2-proxy service definition. This will be used to encrypt the cookies that oauth2-proxy sets for the users.
- You need to restart your oauth2-proxy service to apply the changes. You can use the command `docker-compose restart oauth2-proxy` to do this.

After completing these steps, you should be able to connect oauth2.0 to GitHub. When you visit the URL of your nginx service, you will be redirected to GitHub to authorize your OAuth app. After authorizing, you will be able to access your frontend application.

<https://docs.github.com/en/apps/oauth-apps/building-oauth-apps/authorizing-oauth-apps>

``` bash

FROM continuumio/miniconda3
ARG conda_env=mro_env
ADD /src/environment.yml /src/environment.yml
RUN conda env create -f /src/environment.yml
ENV PATH /opt/conda/envs/$conda_env/bin:$PATH
ENV CONDA_DEFAULT_ENV $conda_env
CMD [ "python", "test.py" ]


```

## DVC installation:

<https://iterative.ai/blog/azure-remotes-in-dvc#:~:text=First%2C%20add%20DVC%20as%20a%20requirement%20to%20your,to%20start%20versioning%20your%20data%20and%20tracking%20experiments>.

I can help you with that. Here is a guide to install and use DVC and git with Azure blob storage:

1. Install DVC with the Azure option: `pip install 'dvc[azure]'`
2. Initialize DVC in your project: `dvc init`
3. Add your large files or directories to DVC: `dvc add path-to-large-file`
4. Commit the changes to git: `git add path-to-large-file.dvc .gitignore` and `git commit -m "Add data file"`
5. Set up your Azure container name and connection string as environment variables: `AZURE_STORAGE_CONTAINER_NAME` and `AZURE_STORAGE_CONNECTION_STRING`
6. Configure DVC to use Azure as the remote storage: `dvc remote add -d Data azure://interim/MLOPS_template_data`
7. Push your data to the remote storage: `dvc push`
8. Pull your data from the remote storage: `dvc pull`

https://dvc.org/doc/user-guide/data-management/remote-storage/azure-blob-storage

https://github.com/josecelano/data-version-control/blob/master/docs/azure-blob-storage.md

MLOPS_cnc_data_temp


az config set defaults.account=
az config set defaults.connection_string="--------------"

dvc remote add -d Data azure://interim/cnc_tmeplate
dvc remote modify Data account_name 

dvc remote modify --local Data connection_string 

dvc config core.hardlink_lock true

