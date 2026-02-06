from abc import abstractmethod
from typing import Any, Dict, List, Union, Literal
from pydantic import BaseModel
import numpy as np

from vmppy.data_type import ViaPoint
from vmppy.transformations import quaternion_slerp


class ElementaryTraj():
    def __init__(self, dim: int) -> None:
        self.h_params = None
        self.dim = dim
        self.virtual_viapoints: List[ViaPoint] = [
            ViaPoint(1.0, np.zeros(dim)),
            ViaPoint(0.0, np.zeros(dim))
        ]
        self.original_viapoints: List[ViaPoint] = [
            ViaPoint(1.0, np.zeros(dim)),
            ViaPoint(0.0, np.zeros(dim))
        ]

    @abstractmethod
    def at(self, phase: float) -> np.ndarray:
        pass

    @abstractmethod
    def rollout(self, can_values: Union[np.ndarray, List[float]]) -> np.ndarray:
        """Get the points at specific phase values.

        Args:
            can_values (Union[np.ndarray, List[float]]): Phase values
                it can be (num_samples,) or (num_samples, 1)
        Returns:
            np.ndarray: Points at the phase values (num_samples, dim)
        """

        return np.array([self.at(phase) for phase in can_values])

    def learn(self, virtual_viapoints: List[ViaPoint]) -> None:
        assert len(virtual_viapoints) >= 2, "At least two virtual viapoints are required"

        # sorted virtual viapoints by can_value
        virtual_viapoints = sorted(virtual_viapoints, key=lambda v: -v.can_value)
        self.virtual_viapoints = virtual_viapoints

    def rollout_with_viapoints(
        self,
        can_values: Union[np.ndarray, List[float]],
        virtual_viapoints: List[ViaPoint]
    ) -> np.ndarray:
        self.learn(virtual_viapoints)
        return self.rollout(can_values)


class LinearElementaryTraj(ElementaryTraj):
    def __init__(self, dim: int) -> None:
        super().__init__(dim)

    def at(self, phase: float) -> np.ndarray:
        """Get the point at a specific phase value.

        Args:
            phase (float): Phase value
        Returns:
            np.ndarray: Point at the phase value (dim,)
        """

        # Find the two closest virtual viapoints
        idx = np.searchsorted([-v.can_value for v in self.virtual_viapoints], -phase)
        if idx == 0:
            return self.virtual_viapoints[idx].point

        # Linear interpolation
        prev = self.virtual_viapoints[idx - 1]
        next = self.virtual_viapoints[idx]
        alpha = (phase - prev.can_value) / (next.can_value - prev.can_value)
        return prev.point + alpha * (next.point - prev.point)


class MinimumJerkElementaryTraj(ElementaryTraj):
    def __init__(self, dim: int) -> None:
        super().__init__(dim)

    def at(self, phase: float) -> np.ndarray:
        idx = np.searchsorted([-v.can_value for v in self.virtual_viapoints], -phase)
        if idx == 0:
            return self.virtual_viapoints[idx].point

        prev = self.virtual_viapoints[idx - 1]
        next = self.virtual_viapoints[idx]
        alpha = (phase - prev.can_value) / (next.can_value - prev.can_value)
        jerk_coeff = (10 * alpha**3 - 15 * alpha**4 + 6 * alpha**5)
        return prev.point + jerk_coeff * (next.point - prev.point)



class TaskSpaceLinearElementaryTraj(ElementaryTraj):
    def __init__(self) -> None:
        super().__init__(dim=7)

    def at(self, phase: float) -> np.ndarray:
        """Get the point at a specific phase value.

        Args:
            phase (float): Phase value
        Returns:
            np.ndarray: Point at the phase value (7,)
             x, y, z, qw, qx, qy, qz
        """
        # Find the two closest virtual viapoints
        idx = np.searchsorted([-v.can_value for v in self.virtual_viapoints], -phase)
        if idx == 0:
            return self.virtual_viapoints[idx].point

        # Linear interpolation for position
        prev = self.virtual_viapoints[idx - 1]
        next = self.virtual_viapoints[idx]
        alpha = (phase - prev.can_value) / (next.can_value - prev.can_value)
        position = prev.point[:3] + alpha * (next.point[:3] - prev.point[:3])

        # Slerp for orientation
        prev_quat = prev.point[3:]
        next_quat = next.point[3:]

        orientation = quaternion_slerp(prev_quat, next_quat, alpha)
        return np.concatenate([position, orientation])


class TaskSpaceMinJerkElementaryTraj(ElementaryTraj):
    def __init__(self) -> None:
        super().__init__(dim=7)

    def at(self, phase: float) -> np.ndarray:
        """Get the point at a specific phase value using minimum jerk for position and slerp for orientation.

        Args:
            phase (float): Phase value between 0 and 1.
        Returns:
            np.ndarray: Point at the phase value (7,)
             x, y, z, qw, qx, qy, qz
        """
        # Find the two closest virtual viapoints
        idx = np.searchsorted([-v.can_value for v in self.virtual_viapoints], -phase)
        if idx == 0:
            return self.virtual_viapoints[idx].point

        prev = self.virtual_viapoints[idx - 1]
        next = self.virtual_viapoints[idx]

        # Minimum jerk interpolation for position
        alpha = (phase - prev.can_value) / (next.can_value - prev.can_value)
        jerk_coeff = (10 * alpha**3 - 15 * alpha**4 + 6 * alpha**5)
        position = prev.point[:3] + jerk_coeff * (next.point[:3] - prev.point[:3])

        # Slerp for orientation
        prev_quat = prev.point[3:]
        next_quat = next.point[3:]
        orientation = quaternion_slerp(prev_quat, next_quat, jerk_coeff)

        return np.concatenate([position, orientation])


class ElementaryTrajConfig(BaseModel):
    type: Literal["linear", "minimum_jerk", "task_space_linear", "task_space_minimum_jerk"] = "linear"
    dim: int = 1


class ElementaryTrajectoryFactory:
    @staticmethod
    def get_elementary_trajectory(config: ElementaryTrajConfig) -> ElementaryTraj:
        if config.type == "linear":
            return LinearElementaryTraj(config.dim)
        elif config.type == "minimum_jerk":
            return MinimumJerkElementaryTraj(config.dim)
        elif config.type == "task_space_linear":
            return TaskSpaceLinearElementaryTraj()
        elif config.type == "task_space_minimum_jerk":
            return TaskSpaceMinJerkElementaryTraj()


def test():
    import matplotlib.pyplot as plt

    # Define virtual viapoints for testing (2D for non-task-space, 7D for task-space)
    virtual_viapoints_2d = [
        ViaPoint(0.0, np.array([0.0, 0.0])),
        ViaPoint(1.0, np.array([1.0, 1.0]))
    ]
    virtual_viapoints_7d = [
        ViaPoint(0.0, np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])),  # Identity quaternion
        ViaPoint(1.0, np.array([1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0]))   # Arbitrary quaternion
    ]
    can_values = np.linspace(0, 1, 100)  # 100 phases between 0 and 1 for smooth trajectory

    # Instantiate Linear and Minimum Jerk Trajectories (2D)
    traj_linear = ElementaryTrajectoryFactory.get_elementary_trajectory(type="linear", dim=2)
    traj_min_jerk = ElementaryTrajectoryFactory.get_elementary_trajectory(type="minimum_jerk", dim=2)

    # Learn the virtual viapoints
    traj_linear.learn(virtual_viapoints_2d)
    traj_min_jerk.learn(virtual_viapoints_2d)

    # Rollout the trajectories
    output_linear = traj_linear.rollout(can_values)
    output_min_jerk = traj_min_jerk.rollout(can_values)

    # Plotting the results for 2D trajectories (time vs each dimension)
    plt.figure(figsize=(10, 5))

    # Plot for the first dimension
    plt.subplot(1, 2, 1)
    plt.plot(can_values, output_linear[:, 0], label='Linear', color='b')
    plt.plot(can_values, output_min_jerk[:, 0], label='Minimum Jerk', color='g')
    plt.title('2D Trajectory - Dimension 1 (X)')
    plt.xlabel('Phase')
    plt.ylabel('X')
    plt.legend()

    # Plot for the second dimension
    plt.subplot(1, 2, 2)
    plt.plot(can_values, output_linear[:, 1], label='Linear', color='b')
    plt.plot(can_values, output_min_jerk[:, 1], label='Minimum Jerk', color='g')
    plt.title('2D Trajectory - Dimension 2 (Y)')
    plt.xlabel('Phase')
    plt.ylabel('Y')
    plt.legend()

    # Display the 2D plots
    plt.tight_layout()
    plt.show()

    # Task-space trajectories are 7D, but we'll only plot positions (first 3 dimensions)
    traj_task_linear = ElementaryTrajectoryFactory.get_elementary_trajectory(type="task_space_linear")
    traj_task_min_jerk = ElementaryTrajectoryFactory.get_elementary_trajectory(type="task_space_minimum_jerk")

    # Learn the virtual viapoints
    traj_task_linear.learn(virtual_viapoints_7d)
    traj_task_min_jerk.learn(virtual_viapoints_7d)

    # Rollout the task-space trajectories
    output_task_linear = traj_task_linear.rollout(can_values)
    output_task_min_jerk = traj_task_min_jerk.rollout(can_values)

    # Plotting the results for Task-space trajectories (time vs each dimension for position x, y, z)
    fig, ax = plt.subplots(3, 1, figsize=(10, 10))

    # Task-space linear trajectory vs time for X, Y, Z
    ax[0].plot(can_values, output_task_linear[:, 0], label='Linear', color='b')
    ax[0].plot(can_values, output_task_min_jerk[:, 0], label='Minimum Jerk', color='g')
    ax[0].set_title('Task-Space Trajectory - Dimension 1 (X)')
    ax[0].set_xlabel('Phase')
    ax[0].set_ylabel('X')

    ax[1].plot(can_values, output_task_linear[:, 1], label='Linear', color='b')
    ax[1].plot(can_values, output_task_min_jerk[:, 1], label='Minimum Jerk', color='g')
    ax[1].set_title('Task-Space Trajectory - Dimension 2 (Y)')
    ax[1].set_xlabel('Phase')
    ax[1].set_ylabel('Y')

    ax[2].plot(can_values, output_task_linear[:, 2], label='Linear', color='b')
    ax[2].plot(can_values, output_task_min_jerk[:, 2], label='Minimum Jerk', color='g')
    ax[2].set_title('Task-Space Trajectory - Dimension 3 (Z)')
    ax[2].set_xlabel('Phase')
    ax[2].set_ylabel('Z')

    for axis in ax:
        axis.legend()

    # Display the task-space plots
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    test()