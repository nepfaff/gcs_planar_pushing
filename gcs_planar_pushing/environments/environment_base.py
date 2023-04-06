from abc import ABC, abstractmethod

from gcs_planar_pushing.controllers import ControllerBase


class EnvironmentBase(ABC):
    """The environment base class."""

    def __init__(self, controller: ControllerBase):
        self._controller = controller

        self._simulation_time = 0.0

    def setup_environment(self) -> None:
        """Sets up the Drake environment."""
        raise NotImplementedError

    @abstractmethod
    def simulate(self, dt: float) -> None:
        """Simulate the environment for `dt` seconds."""
        raise NotImplementedError
