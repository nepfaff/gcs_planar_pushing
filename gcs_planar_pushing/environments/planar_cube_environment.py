from .environment_base import EnvironmentBase


class PlanarCubeEnvironment(EnvironmentBase):
    def __init__(self, controller):
        super().__init__(controller)

    def setup_environment(self) -> None:
        super().setup_environment()

    def simulate(self, dt: float) -> None:
        super().simulate(dt)
