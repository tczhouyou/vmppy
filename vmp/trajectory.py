"""Minimal trajectory utilities for vmppy.

We intentionally keep this lightweight: only what vmp/tvmp need.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np


@dataclass
class Trajectory:
    data: np.ndarray  # (T, D)
    timestamps: np.ndarray  # (T,)

    def __post_init__(self):
        self.data = np.asarray(self.data)
        if self.data.ndim == 1:
            self.data = self.data[:, None]
        if self.timestamps is None:
            self.timestamps = np.arange(len(self.data))
        self.timestamps = np.asarray(self.timestamps)
        if len(self.data) != len(self.timestamps):
            raise ValueError("data and timestamps must have same length")

    @classmethod
    def from_array(cls, data: np.ndarray, timestamps: Optional[np.ndarray] = None) -> "Trajectory":
        if timestamps is None:
            timestamps = np.arange(len(data))
        return cls(np.asarray(data), np.asarray(timestamps))

    @property
    def x0(self) -> np.ndarray:
        return self.data[0]

    @property
    def xT(self) -> np.ndarray:
        return self.data[-1]


class Trajectories(list):
    """A thin container for multiple trajectories."""

    def __init__(self, trajs: Optional[Iterable[Trajectory]] = None):
        super().__init__(trajs or [])


class TSTrajectory(Trajectory):
    """Time-stamped trajectory alias used in the original code."""

    @classmethod
    def from_data(cls, data: np.ndarray, timestamps: Optional[Sequence[float]] = None) -> "TSTrajectory":
        if timestamps is None:
            timestamps = np.arange(len(data))
        return cls(np.asarray(data), np.asarray(timestamps))
