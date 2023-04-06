"""A script for running a single experiment."""

import pathlib

import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath("../..", "config")),
)
def main(cfg: OmegaConf):
    environment = instantiate(cfg.environment)


if __name__ == "__main__":
    main()
