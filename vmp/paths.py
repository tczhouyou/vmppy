"""Local path helpers.

This provides a small `VMPPath` helper without requiring project-wide env vars.

- If VMPPY_ROOT is set, use it.
- Otherwise default to the package directory.
"""

from __future__ import annotations

import os
from os.path import join


def _root() -> str:
    return os.environ.get("VMPPY_ROOT", os.path.dirname(__file__))


class VMPPath:
    ROOT = _root()
    MOTION_LIB_PATH = join(ROOT, "data", "motion_library")
    VMP_MOTION_PATH = join(MOTION_LIB_PATH, "vmp_motions")
    RAW_MOTION_PATH = join(VMP_MOTION_PATH, "raw")

    @staticmethod
    def get_vmp_motion(vmp_name: str, ext: str = "pkl") -> str:
        return join(VMPPath.VMP_MOTION_PATH, f"{vmp_name}.{ext}")

    @staticmethod
    def get_raw_motion(motion_name: str, ext: str = "npy") -> str:
        return join(VMPPath.RAW_MOTION_PATH, f"{motion_name}.{ext}")
