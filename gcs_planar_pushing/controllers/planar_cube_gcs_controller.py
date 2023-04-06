from .controller_base import ControllerBase

from pydrake.all import DiagramBuilder, MultibodyPlant


class PlanarCubeGCSController(ControllerBase):
    """An open-loop GCS controller."""

    def __init__(self):
        super().__init__()

    def setup(self, builder: DiagramBuilder, plant: MultibodyPlant) -> None:
        super().setup(builder, plant)
