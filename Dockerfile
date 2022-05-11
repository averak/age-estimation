FROM nvcr.io/nvidia/tensorflow:22.04-tf2-py3

ENV WORKDIR /app/

WORKDIR ${WORKDIR}

COPY . $WORKDIR

RUN pip install --upgrade pip &&  \
	pip install pipenv && \
	pipenv install --system
