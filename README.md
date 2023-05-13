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

### Running Diffusion Policy

A pre-trained diffusion policy controller can be run as follows:
```
python scripts/run_demo.py --config-name basic.yaml controller=diffusion_policy \
environment.initial_box_position='[2,0]' environment.initial_finger_position='[3,3]'
```

Running with random disturbances:
```
python scripts/run_demo.py --config-name basic.yaml controller=diffusion_policy \
environment.initial_box_position='[-1,2]' environment.initial_finger_position='[3,3]' \
environment.disturbance_probability_per_timestep=0.005 environment.disturbances_max_number=3
```