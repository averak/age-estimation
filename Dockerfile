FROM tensorflow/tensorflow:2.3.0-gpu

ENV WORKDIR /app/

WORKDIR ${WORKDIR}

COPY Pipfile $WORKDIR

RUN pip install --upgrade pip &&  \
	pip install pipenv && \
	pipenv install --system --skip-lock

COPY src $WORKDIR/src

# 利用するdockerイメージのpythonバージョンが3.6なので、
# Python 3.9以降用の型アノテーションを削除する必要がある
RUN grep -l ' \-> list\[.*\]' ./**/*.py | xargs sed -i.bak -e 's/ \-> list\[.*\]//g' && \
	grep -l ': list\[.*\],' ./**/*.py | xargs sed -i.bak -e 's/: list\[.*\],/,/g' && \
	grep -l ': list\[.*\] ' ./**/*.py | xargs sed -i.bak -e 's/: list\[.*\] //g'
