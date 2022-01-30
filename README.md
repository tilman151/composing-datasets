# Compose Datasets, Don't Inherit Them

This is the companion repository to [this blog post](https://krokotsch.eu/posts/compose-datasets/).
It illustrates the design pattern *composition over inheritance* on a PyTorch datasets.

The post references several versions of this repository.
Each version is marked with a git tag:

| Tag      | Description                                                  |
|----------|--------------------------------------------------------------|
| v0.1.0   | Single class for hate speech dataset with fixed tokenizer    |
| v0.2.0   | Hate speech dataset with string argument to choose tokenizer |
| v0.3.0   | Imdb dataset added through a super class                     |
| v0.3.1-a | Revtok tokenizer configurable through **kwargs               |
| v0.3.1-b | Revtok tokenizer configurable through own child class        |
| v0.4.0   | All tokenizers configurable through composition              |

## Installation

First, checkout the repository:

```shell
git clone git@github.com:tilman151/composing-datasets.git
# or
git clone https://github.com/tilman151/composing-datasets.git
```

This project uses *poetry* for dependency management.
Please refer to the [poetry docs](https://python-poetry.org/docs/) for installation instructions.
After installing poetry, install the dependencies with:

```shell
poetry install
```

Poetry will create a clean virtual environment for this project which can be activated with:

```shell
poetry shell
```

## Choosing a Version

Each version is tested and functional.
To choose a specific version, look up the tag in the table above and check the commit out:

```shell
git checkout tags/<version_tag>
```

To verify your installation and the version, run the tests:

```shell
python -m unittest -v
```
