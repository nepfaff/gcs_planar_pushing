from distutils.core import setup

setup(
    name="gcs_planar_pushing",
    version="1.0.0",
    packages=["gcs_planar_pushing"],
    install_requires=[
        "numpy",
        "matplotlib",
        "ipywidgets",
        "pre-commit",
        "manipulation",
        "underactuated",
        "black",
        "hydra-core",
        "omegaconf",
        "wandb",
    ],
)
