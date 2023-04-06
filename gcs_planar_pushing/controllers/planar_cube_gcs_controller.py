from .controller_base import ControllerBase


class PlanarCubeGCSController(ControllerBase):
    """An open-loop GCS controller."""

    def __init__(self):
        super().__init__()
