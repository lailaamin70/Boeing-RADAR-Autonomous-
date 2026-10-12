"""Stand-ins for truckscenes-devkit and pyquaternion, installed only if missing."""
import sys
import types


class _Quaternion:
    """The subset of pyquaternion.Quaternion the app uses: rotation_matrix, inverse, elements."""

    def __init__(self, *args):
        if len(args) == 1:
            args = tuple(args[0])
        w, x, y, z = (float(v) for v in args)
        n = (w * w + x * x + y * y + z * z) ** 0.5
        self.w, self.x, self.y, self.z = w / n, x / n, y / n, z / n

    @property
    def rotation_matrix(self):
        import numpy as np

        w, x, y, z = self.w, self.x, self.y, self.z
        return np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ])

    @property
    def inverse(self):
        return _Quaternion([self.w, -self.x, -self.y, -self.z])

    @property
    def elements(self):
        import numpy as np

        return np.array([self.w, self.x, self.y, self.z])


def install():
    try:
        import truckscenes  # noqa: F401
    except ImportError:
        pkg = types.ModuleType("truckscenes")
        pkg.TruckScenes = object
        utils = types.ModuleType("truckscenes.utils")
        data_classes = types.ModuleType("truckscenes.utils.data_classes")
        data_classes.LidarPointCloud = object
        data_classes.RadarPointCloud = object
        sys.modules.update({
            "truckscenes": pkg,
            "truckscenes.utils": utils,
            "truckscenes.utils.data_classes": data_classes,
        })

    try:
        import pyquaternion  # noqa: F401
    except ImportError:
        mod = types.ModuleType("pyquaternion")
        mod.Quaternion = _Quaternion
        sys.modules["pyquaternion"] = mod
