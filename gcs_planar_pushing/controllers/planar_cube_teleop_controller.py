from .controller_base import ControllerBase


class PlanarCubeTeleopController(ControllerBase):
    """An open-loop teleop controller."""

    def __init__(self):
        super().__init__()
