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

You can run this application from `src/main.py`.

```bash
# record your voice
$ pipenv run python src/main.py --train
```

Please refer to the help for more detailed usage.

```
$ pipenv run python src/main.py --help
usage: main.py [-h] [-r] [-c]

optional arguments:
  -h, --help   show this help message and exit
  -t, --train  学習
```
