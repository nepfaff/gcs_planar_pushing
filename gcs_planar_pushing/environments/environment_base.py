from abc import ABC, abstractmethod

from gcs_planar_pushing.controllers import ControllerBase


class EnvironmentBase(ABC):
    """The environment base class."""

    def __init__(
        self,
        controller: ControllerBase,
        sim_time_step: float,
        scene_directive_path: str,
    ):
        self._controller = controller
        self._time_step = sim_time_step
        self._scene_directive_path = scene_directive_path

        self._simulation_time = 0.0

    def setup(self) -> None:
        """Sets up the Drake environment."""
        raise NotImplementedError

    @abstractmethod
    def simulate(self) -> None:
        """Simulate the environment."""
        raise NotImplementedError
