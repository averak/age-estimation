FROM python:3.9

ENV WORKDIR /app/

WORKDIR ${WORKDIR}

COPY . $WORKDIR

RUN pip install --upgrade pip &&  \
	pip install pipenv && \
	pipenv install
