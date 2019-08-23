FROM drtools/dask:latest

ADD requirements.txt /opt/model/
WORKDIR /opt/model

RUN pip install -r /opt/model/requirements.txt

USER root

ADD . /opt/model


