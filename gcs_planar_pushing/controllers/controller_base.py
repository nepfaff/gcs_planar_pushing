from abc import ABC, abstractmethod
from typing import List

from pydrake.all import DiagramBuilder, MultibodyPlant, Meshcat


class ControllerBase(ABC):
    """The controller base class."""

    def __init__(self, time_step: float):
        self._time_step = time_step
        self._sim_duration = None

    def get_sim_duration(self):
        if self._sim_duration is None:
            raise Exception(
                "Need to set sim_duration before calling get_sim_duration()"
            )
        return self._sim_duration

    @abstractmethod
    def setup(self, builder: DiagramBuilder, plant: MultibodyPlant, **kwargs) -> None:
        """Setup the controller."""
        raise NotImplementedError

    def add_meshcat(self, meshcat: Meshcat) -> None:
        self._meshcat = meshcat

    def set_initial_state(
        self, initial_box_position: List[float], initial_finger_position: List[float]
    ):
        self._initial_box_position = initial_box_position
        self._initial_finger_position = initial_finger_position
