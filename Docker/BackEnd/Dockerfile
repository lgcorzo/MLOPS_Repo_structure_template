FROM continuumio/miniconda3:latest

COPY ../../ /BackEnd
WORKDIR /BackEnd

COPY ./entrypoint.sh ./entrypoint.sh
RUN sudo chmod +x ./entrypoint.sh

# See this tutorial for details https://pythonspeed.com/articles/activate-conda-dockerfile/
RUN conda env create -f Settings/Code/code_development_env.yaml
ENTRYPOINT ["./entrypoint.sh"]


