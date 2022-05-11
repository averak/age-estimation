# Age Estimation

[![MIT License](http://img.shields.io/badge/license-MIT-blue.svg?style=flat)](LICENSE)

Keras implementation of a CNN network for age estimation.

## Requirement

- Python 3.9
- pipenv

## Development

### Installation

```bash
$ pipenv install
```

### Code check & format

```bash
# code check
$ pipenv run lint && pipenv run mypy

# code format
$ pipenv run format
```

## Usage

You need to download [UTKFace](https://susanqq.github.io/UTKFace/) and place the data in `data` directory.

And then, you can run this application from `src/main.py`.

```bash
$ pipenv run python src/main.py --train
```

Please refer to the help for more detailed usage.

```
$ pipenv run python src/main.py --help
usage: main.py [-h] [-r] [-c]

optional arguments:
  -h, --help   show this help message and exit
  -t, --train  学習
  -c, --clear  チェックポイントを削除
```

## Usage with Docker

```bash
$ docker build -t age-estimation .
$ docker run -it age-estimation pipenv run python src/main.py --train
```
