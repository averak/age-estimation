FROM nvcr.io/nvidia/tensorflow:22.04-tf2-py3

ENV WORKDIR /app/

WORKDIR ${WORKDIR}

COPY . $WORKDIR

# 利用するdockerイメージのpythonバージョンが最新でも3.8なので、
# Python 3.9以降用の型アノテーションを削除する
RUN grep -l ' \-> list\[.*\]' ./**/*.py | xargs sed -i.bak -e 's/ \-> list\[.*\]//g' && \
	grep -l ': list\[.*\] ' ./**/*.py | xargs sed -i.bak -e 's/: list\[.*\] //g'

RUN pip install --upgrade pip &&  \
	pip install pipenv && \
	pipenv install --system --skip-lock
