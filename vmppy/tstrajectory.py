from typing import Optional, List
import numpy as np
np.bool = np.bool_

from scipy.interpolate import interp1d
from vmppy.transformations import quaternion_slerp
from vmppy.trajectory import Trajectory


class TSTrajectory(Trajectory):
    def at(self, timestep: float, interp_mode: Optional[str] = None) -> np.ndarray:
        if interp_mode is None:
            interp_mode = self.interp_mode

        if interp_mode == "interp1d":
            fx = interp1d(self.timestamps, self.data[:, :3], axis=0)

            # find two closest timestamps
            idx = np.searchsorted(self.timestamps, timestep)
            if idx == 0:
                quaternion = self.data[0,3:]
            elif idx == len(self.timestamps):
                quaternion = self.data[-1,3:]
            else:
                quaternion = quaternion_slerp(self.data[idx-1,3:], self.data[idx,3:], (timestep - self.timestamps[idx-1]) / (self.timestamps[idx] - self.timestamps[idx-1]))
            position = fx(timestep)
        else:
            raise NotImplementedError

        return np.array([*position, *quaternion])

    @classmethod
    def interpolate(clc, x0: List[float], xT: List[float], num_samples: int) -> np.ndarray:
        traj = clc([x0, xT], [0, 1], interp_mode="interp1d")
        traj.normalize_timestamps(num_samples=num_samples)
        return traj.data[1:]

    def shrink(self, ratio: float, dims: Optional[List[int]] = None):
        if dims is None:
            dims = list(range(3))

        x0 = self.x0
        for d in self.data:
            d[dims] = x0[dims] + ratio * (d[dims] - x0[dims])


    def normalize_trajectory(self, num_samples: int):
        traj = self.data[:, :3]
        diffs = np.diff(traj, axis=0)

        # Compute distances between consecutive points
        distances = np.linalg.norm(diffs, axis=1)
        total_distance = np.sum(distances)

        # Normalize distances to get relative durations for each segment
        relative_durations = distances / total_distance

        # Calculate the cumulative time (starting at 0 and ending at 1, normalized)
        cumulative_durations = np.concatenate(([0], np.cumsum(relative_durations)))

        # Rescale the cumulative durations to match the original duration (based on total time span)
        total_time = self.timestamps[-1] - self.timestamps[0]
        new_timestamps = cumulative_durations * total_time + self.timestamps[0]

        # Update the timestamps to reflect the new speed
        self.timestamps = new_timestamps

        # Ensure the timestamps are normalized if required by your system
        self.normalize_timestamps(num_samples=num_samples)

    def get_length(self):
        traj = self.data[:, :3]
        diffs = np.diff(traj, axis=0)
        distances = np.linalg.norm(diffs, axis=1)
        return np.sum(distances)
