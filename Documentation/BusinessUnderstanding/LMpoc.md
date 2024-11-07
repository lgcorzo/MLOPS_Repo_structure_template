# Project Merlin LLM

$$\color{red}{IMPORTANT}$$
<span style="color:red"> This table is necessary to update </span>

| Version | name | Release Date | Description |
| ------- |---------| ------------ | ----------- |
| 1.0     | Sabin Luja Hernandez |February 13, 2024 | Initial release |
<!-- PULL_REQUESTS_TABLE -->
<!-- cspell:ignore Databricks LANTEK -->
<!-- cspell:enable -->

## Introduction

El objetivo del proyecto es entrenar con datos propios un modelo estandar como el Microsoft CodeBert

- [RAG](https://python.langchain.com/docs/expression_language/cookbook/retrieval) de un LLM ([Microsoft/CodeBert](https://github.com/microsoft/CodeBERT))

- Utilizando [LangChain](https://www.langchain.com/)
  
- Como usar RAG para responder preguntas basadas en el contexto de una conversación y los documentos recuperados: [Conversational Retrieval Chain](https://api.python.langchain.com/en/latest/chains/langchain.chains.conversational_retrieval.base.ConversationalRetrievalChain.html)
  
- Modelo de lenguaje como [CodeBERT](https://github.com/microsoft/CodeBERT)

- Para [LLM](https://www.bbva.com/es/innovacion/los-llm-modelos-de-lenguaje-que-son-y-como-funcionan/)
  
## Como hacer un RAG utilizando LangChain

Para hacer un RAG (generación aumentada por recuperación) utilizando Langchain para el modelo CodeBERT, puedes seguir estos pasos:


  
- Transformación los documentos de conocimiento en vectores de características usando un modelo de incrustación como Hugging Face Embeddings, CodeBERT. Usaremos Elastic Search para crear un almacén de vectores que te permita recuperar los documentos más similares a una consulta.
  
- Crea una cadena de Langchain que tome una consulta como entrada y devuelva una respuesta generada como salida. La cadena debe incluir los siguientes componentes:
  
  - Un recuperador que use el almacén de vectores para obtener los documentos más relevantes para la consulta.
  
  - Un generador que use el modelo CodeBERT para generar una respuesta a partir de la consulta y los documentos recuperados. Puedes usar el pipeline de Hugging Face para acceder al modelo CodeBERT y configurar sus parámetros.
  
  - Un analizador de salida que convierta la respuesta generada en un formato legible.

### Ejemplo que implementa un RAG usando Langchain y CodeBERT (Generico)

#### Importar las librerías necesarias

```python
    from langchain.document_loaders import HuggingFaceDatasetLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.vectorstores import FAISS
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    from langchain import HuggingFacePipeline
    from langchain.chains import RetrievalGeneration
    from langchain.output_parsers import StrOutputParser

```

#### Cargar los datos como documentos de conocimiento
```python
dataset_name = "databricks/databricks-dolly-15k" # o el nombre de tu conjunto de datos
page_content_column = "context" # o la columna que contiene el contenido de los documentos
loader = HuggingFaceDatasetLoader(dataset_name, page_content_column)
data = loader.load()
```

#### Transformar los documentos en vectores de características
```python
embedding = HuggingFaceEmbeddings("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2") # o el nombre de tu modelo de incrustación
vectorstore = FAISS.from_texts(data, embedding=embedding)
```

#### Crear una cadena de Langchain para el RAG
```python
model_name = "microsoft/codebert-base" # o el nombre de tu modelo CodeBERT
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
generator = HuggingFacePipeline(model=model, tokenizer=tokenizer, task="text2text-generation")
chain = RetrievalGeneration(vectorstore=vectorstore, generator=generator, output_parser=StrOutputParser())
```

#### Probar la cadena con una consulta de ejemplo
```python
query = "como puedo hacer un rag utilizando langchain para el modelo codebert"
response = chain.invoke(query)
print(response)
```

### Obtjetivo del proyecto:

Crear un sistema que pueda comparar CNCs, usando los datos de la carpeta Data como base de conocimiento, utilizando sistemas LLM.
  

## Estado del arte respecto a RAG

El estado del arte en cuanto a RAG (Generación Aumentada por Recuperación) implica el uso de modelos de lenguaje avanzados como CodeBERT y técnicas de recuperación de información para generar respuestas precisas y contextualmente relevantes a partir de un conjunto de documentos de conocimiento. 

El enfoque combina capacidades de búsqueda y generación de lenguaje natural para mejorar la comprensión y la calidad de las respuestas. Algunas características clave del estado del arte en RAG son:

1. *Modelos de Lenguaje Avanzados:* Se utilizan modelos de lenguaje preentrenados de vanguardia, como BERT, RoBERTa o en este caso específico, CodeBERT, que están afinados para tareas específicas, como la generación de código.

2. *Recuperación de Información:* Se implementan técnicas de recuperación de información para identificar y seleccionar los documentos relevantes que contienen la información necesaria para responder a una consulta específica. Esto puede incluir algoritmos de búsqueda vectorial, como FAISS, que ayudan a encontrar documentos similares a una consulta dada.

3. *Generación de Respuestas Contextualmente Relevantes:* Los modelos de generación de lenguaje, como los utilizados en este caso, son capaces de producir respuestas coherentes y contextualmente relevantes basadas en la consulta y los documentos recuperados. Esto implica comprender el contexto de la pregunta y sintetizar una respuesta significativa.

4. *Integración de Componentes de Procesamiento de Lenguaje Natural (NLP):* Se aprovechan pipelines de procesamiento de lenguaje natural para realizar tareas como tokenización, generación de embeddings y análisis de salida, lo que simplifica el desarrollo e implementación de sistemas de RAG.

En resumen, el estado del arte en RAG se basa en la combinación de tecnologías de vanguardia en procesamiento de lenguaje natural y recuperación de información para construir sistemas capaces de proporcionar respuestas precisas y contextuales a una amplia gama de consultas basadas en texto.

### Frontend Ideas:

- Programado en python con dash
- El servidor de la aplicacion va a ser un fast API
- Y el CGI va a ser VUnicorn

## Como hacer un Frontend con Dash y LangChain para LLM 

Un frontend es la parte visual e interactiva de una aplicación web. Para hacer un frontend con Dash y LangChain para LLM, necesitas:

- _Dash:_ Es una herramienta de Python que te permite crear aplicaciones web interactivas usando solo código Python.

- _LangChain:_ Es una herramienta que facilita el uso de modelos de lenguaje de gran tamaño (LLM), como CodeBERT, para generar respuestas a partir de una pregunta y unos documentos de conocimiento.

- _FastAPI y VUnicorn:_ Son dos herramientas de Python que te permiten crear y desplegar un servidor web para la aplicación. 

Los pasos para hacer un frontend con Dash y LangChain para LLM son:

- Crear un proceso de LangChain que use CodeBERT y FAISS para generar respuestas a partir de una pregunta y unos documentos de conocimiento (CNC). Un proceso de LangChain es una secuencia de componentes de NLP que producen una salida a partir de una entrada.
  
- Crear una aplicación de Dash que use el proceso de LangChain como backend y que tenga una interfaz de usuario interactiva para introducir la pregunta y mostrar la respuesta. Una aplicación de Dash se compone de dos partes: el diseño y las devoluciones de llamada. El diseño define cómo se ve la aplicación y las devoluciones de llamada definen cómo se comporta la aplicación.

- Desplegar la aplicación de Dash en un servidor web usando FastAPI y VUnicorn. Un servidor web es un programa que recibe las peticiones de los usuarios y envía las respuestas. FastAPI permite crear una API para la aplicación, mientras que VUnicorn, permite ejecutar la aplicación.

### Ejemplo de codigo

``` python
# Paso 1: Instalación de Dependencias

# pip install dash 
# requests fastapi uvicorn

# Paso 2: Configuración del Servidor

from fastapi import FastAPI
import uvicorn

app = FastAPI()

# Paso 3: Crear el Frontend con Dash

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import requests

# Inicializar la aplicación Dash
dash_app = dash.Dash(__name__, server=app, url_base_pathname='/')

# Definir el diseño de la aplicación
dash_app.layout = html.Div([
    html.H1("LangChain for LLM"),
    dcc.Textarea(
        id='input-text',
        placeholder='Ingrese texto aquí...',
        style={'width': '100%', 'height': 200}
    ),
    html.Button('Procesar', id='submit-button', n_clicks=0),
    html.Div(id='output-container')
])

# Definir la lógica de la aplicación
@dash_app.callback(
    Output('output-container', 'children'),
    [Input('submit-button', 'n_clicks')],
    [dash.dependencies.State('input-text', 'value')]
)
def update_output(n_clicks, input_text):
    if n_clicks > 0:
        # Enviar solicitud al servidor FastAPI
        response = requests.post('http://localhost:8000/process', json={'text': input_text})
        output = response.json()['result']
        return html.Div(output)

# Paso 4: Iniciar el Servidor FastAPI

@app.post("/process")
async def process_text(text: str):
    # Aquí se implementa el procesamiento de texto con LangChain para LLM
    processed_text = text.upper()  # Ejemplo: texto en mayúsculas
    return {"result": processed_text}

# Paso 5: Ejecutar el Servidor Dash
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

```