# Age Estimation

[![MIT License](http://img.shields.io/badge/license-MIT-blue.svg?style=flat)](LICENSE)

Keras implementation of a CNN network for age estimation.

## Requirement

- Python 3.9
- pipenv

## Development

### Installation

```bash
$ pipenv install --dev
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
  -t, --train           学習
  -e, --estimate        推定
```

### Run on docker

After downloading the dataset, execute the following command.

```bash
# build docker image
$ docker build -t age-estimation .

# train
$ docker run --gpus=all \
  -v /home/abe/age-estimation/data:/app/data \
  -v /home/abe/age-estimation/ckpt:/app/ckpt \
  -v /home/abe/age-estimation/analysis:/app/analysis \
  -it age-estimation python src/main.py --train

# estimate & make heatmap
$ docker run --gpus=all \
  -v /home/abe/age-estimation/data:/app/data \
  -v /home/abe/age-estimation/ckpt:/app/ckpt \
  -v /home/abe/age-estimation/analysis:/app/analysis \
  -it age-estimation python src/main.py --estimate
```
