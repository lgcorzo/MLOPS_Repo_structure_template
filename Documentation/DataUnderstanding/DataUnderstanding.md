# Project Merlin LLM

$$\color{red}{IMPORTANT}$$
<span style="color:red"> This table is necessary to update </span>

| Version | name | Release Date | Description |
| ------- |---------| ------------ | ----------- |
| 1.0     | Sabin Luja Hernandez |February 19, 2024 | Initial release |
<!-- PULL_REQUESTS_TABLE -->
<!-- cspell:ignore Databricks LANTEK -->
<!-- cspell:enable -->

## Introduction
- [Resumen del documento](https://dev.azure.com/lanteksms/Merlin/_wiki/wikis/Merlin.wiki/1914/BusinessUnderstanding?anchor=b.)
- [Estructura de CNC](https://www.programacioncnc.es/la-estructura-de-un-programa-de-cnc/)
- Tipos de CNC
- Arquitectura del programa 
  - BackEnd
    - LangChain
    - MLFlow
    - DVC
    - Fast API
  - FrontEnd
    - Dash 
    - Fast API
  - MongoDB
  - ElasticSearch

- Mejor forma para usar langchain, codebert
- Mejor forma para ejecutar LLM en CNCs

## Resumen del documento
Resumen de las secciones (Análisis de datos - Estructura de un archivo CNC - Procesamiento de los archivos CNC con Python)

- _Análisis de datos:_ Se analizan los archivos CNC proporcionados por Lantek, revelando información clave como la variedad de extensiones, la longitud de los archivos, la frecuencia de los procesadores posteriores y la diversidad de las piezas. Estos hallazgos ayudan a diseñar una estrategia de preprocesamiento y división de datos para evitar el sobreajuste y mejorar el rendimiento del modelo.

- _Estructura de un archivo CNC:_ Los archivos CNC se componen de varios componentes, como coordenadas, comentarios y códigos. La limpieza de los datos implica la eliminación de la información específica de la pieza de los archivos CNC1. Los archivos CNC se dividen en tres partes: encabezado, cuerpo y cola. Entre estas partes, el encabezado y la cola contienen información sobre el procesador posterior, mientras que el cuerpo es específico de la pieza que se construye.

- _Procesamiento de los archivos CNC con Python:_ Se aplican varios pasos de procesamiento a cada parte del archivo CNC, como la eliminación de valores numéricos, caracteres especiales, líneas duplicadas y ejes seguidos de valores numéricos. El objetivo es eliminar el ruido y conservar la información relevante para el procesador posterior.

## Estructura de CNC
Una máquina CNC consta de seis elementos principales:

- Dispositivo de entrada: Es el medio por el cual se introducen las instrucciones en la máquina.

- Unidad de control o controlador: Es el cerebro de la máquina. Interpreta y ejecuta las instrucciones proporcionadas.

- Máquina herramienta: Es la parte de la máquina que realiza el trabajo físico de mecanizado.

- Sistema de accionamiento: Controla el movimiento de la máquina herramienta.

- Dispositivos de realimentación: Estos son necesarios en sistemas con servomotores para proporcionar un control preciso del movimiento.

Un programa CNC se compone de un conjunto de bloques o instrucciones debidamente ordenadas en subrutinas o en el cuerpo del programa, de esta forma se le suministra al CNC toda la información necesaria para el mecanizado de la pieza. La estructura de un programa CNC generalmente consta de tres partes3:

1. Cabecera del programa: Aquí se aloja el nombre o número del programa, la llamada a la herramienta, las rpm de la herramienta, el nombre y corrector de la herramienta.
   
2. Bloques o líneas del programa: Aquí se indican línea a línea a través de la programación CNC los movimientos que se quieren que realice la herramienta: trayectorias, avances, compensaciones de radio, etc.

3. Fin de programa: Se puede definir mediante las funciones M02 ó M30, ambas equivalentes y de uso opcional. Con M02 se para el programa y con M30 se para el programa y se vuelve al inicio del programa.

## Tipos de CNC

Existen varios tipos de máquinas CNC, cada una de las cuales se diferencia en su modo de funcionamiento, herramienta de corte, materiales y número de ejes que pueden cortar simultáneamente. Aquí te presento algunos tipos de máquinas CNC según su función56:

1. Tornos CNC: Utilizados para fabricar objetos cilíndricos y realizar el proceso de producción de piezas de torneado CNC.

2. Fresadoras CNC: Se utilizan para mecanizar superficies planas y formas complejas en una pieza de trabajo.

3. Enrutadores CNC: Se utilizan para mecanizar materiales más blandos y pueden tener una precisión ligeramente menor en comparación con las fresadoras CNC.

4. Cortadoras de plasma CNC: Se utilizan para cortar metales y otros materiales utilizando un chorro de plasma.

5. Rectificadoras CNC: Utilizan una muela abrasiva giratoria para rectificar el material, adoptando la forma deseada.

6. Impresoras 3D CNC: Se utilizan para crear objetos tridimensionales a partir de un modelo digital.