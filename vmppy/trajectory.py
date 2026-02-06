from typing import Dict, Optional, List, Union
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from scipy.interpolate import interp1d, CubicSpline
from scipy.signal import savgol_filter

from vmppy.logging_utils import setup_logging
from vmppy.io import CSVReader, FolderHandler


class Trajectory():
    def __init__(
        self,
        data: Union[np.ndarray, List[List[float]]] = [],
        timestamps: Optional[Union[np.ndarray, List[float]]] = None,
        interp_mode: str = "interp1d",
        dist_mode: str = "dtw"
    ):
        if isinstance(data, list):
            data = np.array(data)
        self.data = data

        if timestamps is None:
            timestamps = np.arange(len(data))

        assert len(data) == len(timestamps), "Data and timestamps must have the same length"
        self.timestamps = np.array(timestamps)
        self.interp_mode = interp_mode
        self.dist_mode = dist_mode
        self.logger = setup_logging("Trajectory")

    @classmethod
    def from_file(cls, file_path: str, contain_timestamps=False):
        if file_path.endswith(".csv") or file_path.endswith(".txt"):
            data = CSVReader(file_path).np_data
        elif file_path.endswith(".npy"):
            data = np.load(file_path)
        else:
            raise ValueError("Invalid file format")

        if contain_timestamps:
            return cls(data[:, 1:], data[:, 0])

        if len(data.shape) == 1: # data should have shape (n_samples, n_dims)
            data = np.expand_dims(data, axis=-1)
        return cls(data)

    def to_file(self, file_path: str, store_timestamps=False):
        if store_timestamps:
            data = np.column_stack((self.timestamps, self.data))
        else:
            data = self.data

        if file_path.endswith(".csv"):
            pd.DataFrame(data).to_csv(file_path, index=False)
        elif file_path.endswith(".npy"):
            np.save(file_path, data)
        else:
            raise ValueError("Invalid file format")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        row, col = key
        if isinstance(row, slice) or isinstance(col, slice):
            return np.array([r[col] for r in self.data[row]])
        else:
            return self.data[row][col]

    def at(self, timestep: float, interp_mode: Optional[str] = None, derivative: int = 0, diff: float = 0.02) -> np.ndarray:
        if interp_mode is None:
            interp_mode = self.interp_mode

        if interp_mode == "interp1d":
            if derivative == 0:
                f = interp1d(self.timestamps, self.data, axis=0, fill_value="extrapolate")
                return f(timestep)

            else:
                if derivative == 1:
                    return (f(timestep + diff) - f(timestep - diff)) / (2 * diff)
                elif derivative == 2:
                    return (f(timestep + diff) - 2 * f(timestep) + f(timestep - diff)) / (diff ** 2)
                else:
                    raise NotImplementedError("Higher-order derivatives not implemented for 'interp1d'.")

        elif interp_mode == "cubic":
            spline = CubicSpline(self.timestamps, self.data, axis=0)
            return spline(timestep, nu=derivative)

        else:
            raise ValueError(f"Interpolation mode '{interp_mode}' is not implemented.")

    def sub_trajectory(self, start: float, end: float):
        start_idx = np.argmin(np.abs(self.timestamps - start))
        end_idx = np.argmin(np.abs(self.timestamps - end))
        return Trajectory(self.data[start_idx:end_idx], self.timestamps[start_idx:end_idx])

    def __add__(self, other):
        assert len(self) == len(other), "Trajectories must have the same length"
        return Trajectory(self.data + other.data, self.timestamps)

    def __sub__(self, other):
        assert len(self) == len(other), "Trajectories must have the same length"
        return Trajectory(self.data - other.data, self.timestamps)

    def concat_data(self, other):
        self.data = np.vstack((self.data, other.data))

    def concat(self, other):
        other_data = np.zeros_like(self.data)
        for i, timestamp in enumerate(self.timestamps):
            other_data[i] = other.at(timestamp)

        self.data = np.vstack((self.data, other_data))

    def append(self, data: Union[List[float], np.ndarray]):
        if len(self.data) != 0:
            assert len(data) == self.data.shape[1], "Data must have the same dimension as the trajectory"

        if isinstance(data, list):
            data = np.array(data)

        if len(self.data) == 0:
            self.data = np.array([data])
        else:
            self.data = np.append(self.data, [data], axis=0)

    def plot(self, ax=None, dims: Optional[List[int]] = None, pivot_dim: Optional[int] = None, name: str = ""):
        if ax is None:
            _, ax = plt.subplots()

        if pivot_dim is not None:
            x = self.data[:, pivot_dim]
            xlabel = f"dim {pivot_dim}"
        else:
            x = self.timestamps
            xlabel = "time"

        if dims is None:
            dims = range(self.data.shape[1])

        for dim in dims:
            ax.plot(x, self.data[:, dim], label=f"{name}-dim {dim}")

        ax.set_xlabel(xlabel)
        ax.set_ylabel(f"Value")
        ax.set_title("Trajectory")
        return ax

    @property
    def xT(self):
        return self.data[-1]

    @property
    def x0(self):
        return self.data[0]

    def normalize_timestamps(self, num_samples: int = 100):
        self.logger.info(
            "Normalizing timestamps."
            "This function will lose the meaning of timestamps."
            "It is recommended to use subsample or change_dt instead."
        )
        self.timestamps = (self.timestamps - self.timestamps[0]) / (self.timestamps[-1] - self.timestamps[0])
        num_samples = max(num_samples, 2)
        timestamps = np.linspace(0, 1, num_samples, endpoint=True)
        data = []
        for i in range(num_samples):
            data.append(self.at(timestamps[i]))

        self.data = np.array(data)
        self.timestamps = timestamps

    def insert(self, timestamp: float, data: Union[List[float], np.ndarray]):
        idx = np.argmin(np.abs(self.timestamps - timestamp))
        self.data = np.insert(self.data, idx, data, axis=0)
        self.timestamps = np.insert(self.timestamps, idx, timestamp)

    def subsample(self, num_samples: int):
        timestamps = np.linspace(self.timestamps[0], self.timestamps[-1], num_samples, endpoint=True)
        data = []
        for i in range(num_samples):
            data.append(self.at(timestamps[i]))

        self.data = np.array(data)
        self.timestamps = timestamps

    def change_dt(self, dt: float = 0.02):
        timestamps = np.arange(self.timestamps[0], self.timestamps[-1], dt)
        data = []
        for i in range(len(timestamps)):
            data.append(self.at(timestamps[i]))

        self.data = np.array(data)
        self.timestamps = timestamps

    def change_speed(self, speed_factor: float):
        self.timestamps = self.timestamps * (1.0 / speed_factor)

    def smooth(self, window_length: int = 5, polyorder: int = 2, mode: str = 'savgol'):
        """
        Smooth the trajectory data.

        Args:
            window_length (int): The length of the smoothing window (must be an odd integer).
            polyorder (int): The order of the polynomial used for the Savitzky-Golay filter.
            mode (str): The smoothing mode ('savgol' or 'moving_average').
        """
        if window_length % 2 == 0:
            raise ValueError("window_length must be an odd integer.")

        if mode == 'savgol':
            self.data = savgol_filter(self.data, window_length, polyorder, axis=0)
        elif mode == 'moving_average':
            def moving_average(x, window_length):
                """ Helper method to compute the moving average. """
                return np.convolve(x, np.ones(window_length)/window_length, mode='same')
            self.data = np.apply_along_axis(moving_average, 0, self.data, window_length)
        else:
            raise ValueError("Invalid mode. Use 'savgol' or 'moving_average'.")

    def extend_ends(self, num_samples: int, time_span_factor: float = 0.1):
        if num_samples <= 0:
            return

        new_data = np.zeros((num_samples, self.data.shape[1]))
        for i in range(num_samples):
            new_data[i] = self.data[-1]

        self.data = np.vstack((self.data, new_data))
        end_time = self.timestamps[-1]
        self.timestamps = np.append(self.timestamps, np.linspace(end_time, end_time * (1 + time_span_factor), num_samples))


class Trajectories():
    def __init__(self, trajectories: Union[List[Trajectory], Dict[str, Trajectory]]=[]) -> None:
        self.num_demos = len(trajectories)
        self.trajectories = trajectories

    def __len__(self):
        return self.num_demos

    def __iter__(self):
        return iter(self.trajectories)

    @classmethod
    def from_files(self, files: List[str]):
        trajectories = []
        for file in files:
            trajectories.append(Trajectory.from_file(file))

        return Trajectories(trajectories)

    @classmethod
    def from_folder(self, folder: str):
        files = FolderHandler.sorted_files_with_numbers(folder)
        return Trajectories.from_files([f"{folder}/{file}" for file in files])

    def plot(self, dims: Optional[List[int]] = None, pivot_dim: Optional[int] = None, all_in_one: bool = False, **kwargs):
        _, ax = plt.subplots()
        for i, trajectory in enumerate(self.trajectories):
            if all_in_one:
                ax = trajectory.plot(ax, dims, pivot_dim, name=f"traj-{i}")
            else:
                _, ax = plt.subplots()
                trajectory.plot(ax, dims, pivot_dim)

        if kwargs.get("legend", False):
            plt.legend()

    def normalize_timestamps(self, num_samples: int = 100):
        for trajectory in self.trajectories:
            trajectory.normalize_timestamps(num_samples)

    def smooth(self, window_length: int = 5, polyorder: int = 2, mode: str = 'savgol'):
        for trajectory in self.trajectories:
            trajectory.smooth(window_length, polyorder, mode)

    def saveall(self, folder: str, store_timestamps=False, file_type="csv"):
        for i, trajectory in enumerate(self.trajectories):
            trajectory.to_file(f"{folder}/traj_{i}.{file_type}", store_timestamps)

    def append(self, trajectory: Trajectory):
        self.trajectories.append(trajectory)
        self.num_demos += 1


if __name__ == "__main__":
    pass