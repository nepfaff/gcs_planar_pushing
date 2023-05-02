"""A script for running a single experiment."""

import pathlib

import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf

from gcs_planar_pushing.environments import EnvironmentBase
from gcs_planar_pushing.controllers import ControllerBase


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath("../..", "config")),
)
def main(cfg: OmegaConf):
    controller: ControllerBase = instantiate(cfg.controller)
    environment: EnvironmentBase = instantiate(cfg.environment, controller=controller)
    environment.setup()
    environment.simulate()


if __name__ == "__main__":
    main()
