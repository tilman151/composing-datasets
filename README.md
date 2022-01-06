# Compose Datasets, Don't Inherit Them

This is the companion repository to [this blog post](http://krokotsch.eu).
It illustrates the design pattern *composition over inheritance* on a PyTorch datasets.

The post references several versions of this repository.
Each version is marked with a git tag:

| Tag | Version |
|-----|---------|
|     |         |

## Installation

This project uses *poetry* for dependency management.
Please refer to the [poetry docs](https://python-poetry.org/docs/) for installation instructions.
After installing poetry, install the dependencies with:

```shell
poetry install
```

If you already activated a virtual environment, poetry will install the dependencies there.
Otherwise, it will create a new one which can be activated with:

```shell
poetry shell
```