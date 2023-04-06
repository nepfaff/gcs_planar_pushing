from abc import ABC, abstractmethod

from pydrake.all import DiagramBuilder, MultibodyPlant


class ControllerBase(ABC):
    """The controller base class."""

    def __init__(self, time_step: float):
        self._time_step = time_step

    @abstractmethod
    def setup(self, builder: DiagramBuilder, plant: MultibodyPlant) -> None:
        """Setup the controller."""
        raise NotImplementedError
