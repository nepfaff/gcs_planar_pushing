# gcs_planar_pushing

## Installation

Clone the repo and execute the following commands from the repository's root.

Install the `gcs_planar_pushing` package in development mode:
```
pip install -e .
```

Install `pre-commit` for automatic black formatting:
```
pre-commit install
```

## Runing a single experiment

Create a config file specifying the experiment in `config` and run it using the following command:

```
python scripts/run_demo.py --config-name basic.yaml
```

where `basic.yaml` should be replaced with your config name.
