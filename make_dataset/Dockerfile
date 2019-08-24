FROM drtools/dask:latest

ADD requirements.txt /opt/make_dataset/
WORKDIR /opt/make_dataset

RUN pip install -r /opt/make_dataset/requirements.txt

USER root

ADD . /opt/make_dataset