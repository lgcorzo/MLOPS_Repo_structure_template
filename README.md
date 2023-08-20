# Project ProjectName

$$\color{red}{IMPORTANT}$$
<span style="color:red"> This table is necessary to update </span>

| Version | name | Release Date | Description |
| ------- |---------| ------------ | ----------- |
| 1.0     | Luis Galo Corzo |February 13, 2023 | Initial release |
<!-- PULL_REQUESTS_TABLE -->
<!-- cspell:ignore Databricks LANTEK -->
<!-- cspell:disable -->

## Contents

- [Contents](#contents)
- [Introduction](#introduction)
- [Scope](#scope)
- [Objectives](#objectives)
- [LLama 2 guides](#llama-2-guides)
- [Requirements](#requirements)
- [Structure of the repo](#structure-of-the-repo)
- [testing](#testing)
- [Keywords](#keywords)
- [Frontend](#frontend)
- [references](#references)
- [mongoDB flask example:](#mongodb-flask-example)
- [end to end  ml project](#end-to-end--ml-project)
- [codebert finetune](#codebert-finetune)
- [diagrams](#diagrams)

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

-Add a service  to call [llama](https://github.com/facebookresearch/llama?fbclid=IwAR0Ngm1SeDDfj6fmSmo-C7e8ERAjUdmD2JvCnR2G_HCez4hFqQw3viCWKOg)

 documentation related
<https://huggingface.co/blog/llama2>

[demo](https://huggingface.co/blog/llama2#demo)
[demo2](https://labs.perplexity.ai/)
inferencing endpoints hugginface [here](https://huggingface.co/docs/inference-endpoints/index)

## LLama 2 guides

[step by step Guide huggingface llama2](https://www.pinecone.io/learn/llama-2/)

[![video](https://markdown-videos.vercel.app/youtube/6iHVJyX2e50)](https://youtu.be/6iHVJyX2e50)

[original paper](https://ai.meta.com/research/publications/llama-2-open-foundation-and-fine-tuned-chat-models/)

[step by step Guide huggingface llama2](https://huggingface.co/blog/llama2)

[huggingface doc](https://huggingface.co/docs/transformers/main/en/model_doc/llama2)

[pipelines huggingface](https://huggingface.co/docs/transformers/v4.17.0/en/pipeline_tutorial)

<https://aitoolmall.com/news/how-to-use-huggingface-llama-2/>

[cpu example](https://github.com/randaller/llama-cpu)

how to use llama with hugggingface

- [One](https://github.com/randaller/llama-cpu)
- [Two](https://huggingface.co/docs/transformers/main/en/model_doc/llama2)

code example:

``` python

from transformers import AutoTokenizer, LlamaForCausalLM

model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

prompt = "Hey, are you conscious? Can you talk to me?"
inputs = tokenizer(prompt, return_tensors="pt")

# Generate
generate_ids = model.generate(inputs.input_ids, max_length=30)
tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
```

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

path to the Frontend http://localhost:9000/frontend-service-dev/

## references
NLP
https://medium.com/analytics-vidhya/best-nlp-algorithms-to-get-document-similarity-a5559244b23b
https://github.com/jairNeto/warren_buffet_letters/tree/main

huggingface repo:
<https://github.com/huggingface/transformers/tree/a5cc30d72ae2dc19af534e4b35c986cc28db1275>

<https://huggingface.co/docs/transformers/task_summary>
 for visio and pipelines
<https://theaisummer.com/hugging-face-vit/>

ProjectName MachineConfiguratorFinder
project_name  example machine_configuration_finder

delete pytest cache

``` powershell
Get-ChildItem -Path . -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force
```

```cmd
rmdir /s /q .pytest_cache

```
Hubert codeBert
I’m sorry, but BERT is not suitable for comparing the documents you provided, because they are not text but code. BERT is a text-based model that relies on word or subword tokens, while your documents are composed of symbols, numbers and abbreviations that BERT cannot understand. You need a model that can handle code or speech data, such as HuBERT or CodeBERT

CodeBert gives better results.



microsoft/CodeBERT: The original repository of CodeBERT that includes code for pre-training, probing and downstream tasks such as code search, code summarization, code documentation generation and code clone detection.

neulab/code-bert-score: CodeBERTScore: an automatic metric for evaluating the quality of generated code using CodeBERT. It can compute the semantic similarity between two code snippets in the same or different programming languages.

microsoft/CodeXGLUE: CodeXGLUE: a benchmark dataset and open challenge for natural language and programming language tasks. It includes 14 tasks that cover various aspects of code understanding and generation. It uses CodeBERT as a pre-trained model for some of the tasks


https://github.com/microsoft/CodeBERT



Hubert example:

``` python
# Import libraries
import torch
from transformers import AutoTokenizer, AutoModel

# Load huBERT model and tokenizer
model = AutoModel.from_pretrained("microsoft/hubert-base-ls960")
tokenizer = AutoTokenizer.from_pretrained("microsoft/hubert-base-ls960")

# Define a function to compute cosine similarity between two vectors
def cosine_similarity(x, y):
  return torch.dot(x, y) / (torch.norm(x) * torch.norm(y))

# Define a function to compare two documents using huBERT
def compare_documents(doc1, doc2):
  # Tokenize and encode the documents
  input_ids1 = tokenizer(doc1, return_tensors="pt").input_ids
  input_ids2 = tokenizer(doc2, return_tensors="pt").input_ids

  # Extract the last hidden state of the [CLS] token
  output1 = model(input_ids1).last_hidden_state[:, 0, :]
  output2 = model(input_ids2).last_hidden_state[:, 0, :]

  # Compute the cosine similarity between the outputs
  similarity = cosine_similarity(output1, output2).item()

  # Return the similarity score
  return similarity

# Test the function with two sample documents
doc1 = "The sky is blue and the sun is shining."
doc2 = "The weather is nice and sunny today."

similarity = compare_documents(doc1, doc2)
print(f"The similarity between the two documents is {similarity:.2f}")
```

Codebert
https://github.com/microsoft/CodeBERT
https://huggingface.co/microsoft/codebert-base

huggingace api.
``` python 
import requests

API_URL = "https://api-inference.huggingface.co/models/microsoft/codebert-base"
headers = {"Authorization": "Bearer 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"}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()
	
output = query({
	"inputs": "Today is a sunny day and I'll get some ice cream.",
})
```

``` python 
# Import libraries
import torch
from transformers import AutoTokenizer, AutoModel

# Load CodeBERT model and tokenizer
model = AutoModel.from_pretrained("microsoft/codebert-base")
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")

# Define a function to compute cosine similarity between two vectors
def cosine_similarity(x, y):
  return torch.dot(x, y) / (torch.norm(x) * torch.norm(y))

# Define a function to compare two documents using CodeBERT
def compare_documents(doc1, doc2):
  # Tokenize and encode the documents
  input_ids1 = tokenizer(doc1, return_tensors="pt").input_ids
  input_ids2 = tokenizer(doc2, return_tensors="pt").input_ids

  # Extract the last hidden state of the [CLS] token
  output1 = model(input_ids1).last_hidden_state[:, 0, :]
  output2 = model(input_ids2).last_hidden_state[:, 0, :]

  # Compute the cosine similarity between the outputs
  similarity = cosine_similarity(output1, output2).item()

  # Return the similarity score
  return similarity

# Test the function with two sample documents
doc1 = "The sky is blue and the sun is shining."
doc2 = "The weather is nice and sunny today."

similarity = compare_documents(doc1, doc2)
print(f"The similarity between the two documents is {similarity:.2f}")

```


``` python
import requests
from transformers import AutoTokenizer

# Load the CodeBERT tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")

# Define a function to compare two documents using the Hugging Face inference API
def compare_documents(doc1, doc2):
  # Encode the documents
  encoded_inputs = tokenizer(text=doc1, text2=doc2, return_tensors="pt")

  # Make an API request to the Hugging Face inference API
  response = requests.post(
    "https://api-inference.huggingface.co/models/microsoft/codebert-base",
    json=encoded_inputs,
  )

  # Get the similarity score from the response
  similarity = response.json()["outputs"]["last_hidden_state"]["0"]["cos_sim"]

  # Return the similarity score
  return similarity

# Test the function with two sample documents
doc1 = "The sky is blue and the sun is shining."
doc2 = "The weather is nice and sunny today."

similarity = compare_documents(doc1, doc2)
print(f"The similarity between the two documents is {similarity:.2f}")

```

installation huggingface trasformers
[transformers](https://huggingface.co/docs/transformers/index)

``` bash

pip install transformers
transformers-cli download microsoft/codebert-base
```


edge gpt api
https://github.com/acheong08/EdgeGPT

 ``` bash
 python3 -m pip install EdgeGPT --upgrade
 ```


 https://www.gradio.app/guides/quickstart
 https://github.com/oobabooga/text-generation-webui

URL para conectar con el ejemplo:
 http://localhost:9000/frontend-service-dev/


## mongoDB flask example:

https://github.com/hoangdesu/Spark-MongoDB-MLflow/blob/main/src/hoang.ipynb

https://mioi.io/blog/posts/docker-mongo/   (mongodb + express + docker compose)

https://www.digitalocean.com/community/tutorials/how-to-set-up-flask-with-mongodb-and-docker
https://github.com/docker/awesome-compose/tree/master/nginx-flask-mongo


## end to end  ml project

https://github.com/shanakaChathu/churn_model
https://medium.com/@shanakachathuranga/end-to-end-machine-learning-pipeline-with-mlops-tools-mlflow-dvc-flask-heroku-evidentlyai-github-c38b5233778c

## codebert finetune

https://colab.research.google.com/github/Ankur3107/colab_notebooks/blob/master/classification/BERT_Fine_Tuning_Sentence_Classification_v2.ipynb

Sure, I can create an example of fine-tuning CodeBERT to calculate similarity between documents using cosine distance. Cosine distance is a measure of how dissimilar two vectors are, based on the angle between them. It is the complement of cosine similarity, which is a measure of how similar two vectors are. Cosine similarity and distance are often used to compare documents in natural language processing and information retrieval.

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

I hope this example helps you understand how to fine-tune CodeBERT for calculating similarity between documents using cosine distance. If you have any questions or feedback, please let me know 😊.

Source: Conversation with Bing, 12/08/2023
(1) How to Calculate Cosine Similarity in Python? - GeeksforGeeks. https://www.geeksforgeeks.org/how-to-calculate-cosine-similarity-in-python/.
(2) sklearn.metrics.pairwise.cosine_similarity — scikit-learn 1.3.0 .... https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html.
(3) Cosine Similarity - Understanding the math and how it works? (with python). https://www.machinelearningplus.com/nlp/cosine-similarity/.


## diagrams

::: mermaid
classDiagram
  class ProjectNameModel {
    -List[Any] knowledge_tokenized
    -List[Any] knowledge_cnc_name
    -Any pretrained_model
    -Any pretrained_tokenizer
    -pd.DataFrame cnc_df
    +__init__(cnc_path) None
    +load_pretrained_llm(llm_type) None
    +fit(x_in, y_in) ProjectNameModel
    +predict_probea(x_in, num_results) pd.DataFrame
    +score(x_test, y_test) float
    +predict(x_in) pd.Series
  }

:::

