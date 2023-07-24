FROM continuumio/miniconda3

RUN pip install dash==2.6.2 \
    && pip install dash-extensions==0.0.13 \
    && pip install cloudpickle \
    && pip install flask==2.2.2 \
    && pip install numpy==1.23.3 \
    && pip install pandas==1.5.0 \
    && pip install requests==2.28.1 \
    && pip install scikit-learn==1.1.2 \
    && pip install tqdm==4.64.0 \
    && pip install python-dotenv==1.0.0 \
    && pip install pyarmor \
    && pip install gunicorn


COPY . /Projects
WORKDIR /Projects

RUN chmod +x start_be.sh
RUN chmod +x start_fe.sh
RUN chmod +x start_app.sh

CMD ["/bin/bash", "-c", "./start_app.sh"]

