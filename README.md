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
