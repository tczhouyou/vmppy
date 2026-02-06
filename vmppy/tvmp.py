import os
import numpy as np
np.bool = np.bool_

from typing import List
import pickle

# Allow both package import (recommended) and direct script execution.
try:
    from .transformations import quaternion_multiply, quaternion_conjugate
    from .trajectory import Trajectories
    from .tstrajectory import TSTrajectory
except ImportError:  # pragma: no cover
    from vmppy.transformations import quaternion_multiply, quaternion_conjugate
    from vmppy.trajectory import Trajectories
    from vmppy.tstrajectory import TSTrajectory


import sys
file_path = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(file_path, "../../")))
sys.path.append(os.path.abspath(os.path.join(file_path, "..")))
sys.path.append(os.path.abspath(file_path))


from vmppy.data_type import ViaPose
from vmppy.vmp import VMP
from vmppy.canonical_system import CanonicalSystemConfig
from vmppy.elementary_traj import ElementaryTrajConfig
from vmppy.function_approximator import FunctionApproximatorConfig


class TVMP(VMP):
    def __init__(
        self,
        shape_modulation_config: FunctionApproximatorConfig = FunctionApproximatorConfig(),
        elementary_traj_config: ElementaryTrajConfig = ElementaryTrajConfig(type="task_space_linear", dim=7),
        canonical_system_config: CanonicalSystemConfig = CanonicalSystemConfig(),
        **kwargs
    ):
        if hasattr(shape_modulation_config.config, "boundary_start"):
            shape_modulation_config.config.boundary_start = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])

        if hasattr(shape_modulation_config.config, "boundary_end"):
            shape_modulation_config.config.boundary_end = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])

        super().__init__(
            dim=7,
            shape_modulation_config=shape_modulation_config,
            elementary_traj_config=elementary_traj_config,
            canonical_system_config=canonical_system_config,
            **kwargs
        )
        self.viapoints: List[ViaPose] = []

    @property
    def vmp_type(self):
        return "tvmp"

    def get_virtual_viapoints(self, viapoints: List[ViaPose]) -> List[ViaPose]:
        virtual_viapoints = []
        for viapoint in viapoints:
            if viapoint.can_value >= 1.0:
                virtual_viapoints.append(viapoint)
            elif viapoint.can_value <= 0.0:
                virtual_viapoints.append(viapoint)
            else:
                fvia = self.shape_modulation(viapoint.can_value).squeeze(0)

                # Calculate the virtual viapoint position
                hposi = viapoint.point[:3] - fvia[:3]

                # Calculate the orientation difference
                vquat = viapoint.point[3:]
                fquat = fvia[3:]
                hquat = quaternion_multiply(vquat, quaternion_conjugate(fquat))

                virtual_viapoints.append(
                    ViaPose.from_position_orientation(viapoint.can_value, hposi, hquat)
                )

        return virtual_viapoints

    def compose(self, fpart: np.ndarray, hpart: np.ndarray) -> np.ndarray:

        fpart = fpart.reshape(-1, 7, order="F")
        hpart = hpart.reshape(-1, 7, order="F")
        res = np.zeros_like(fpart)
        # Calculate the position part
        res[:, :3] = hpart[:, :3] + fpart[:, :3]

        # Calculate the orientation part
        for i in range(len(fpart)):
            if fpart[i, 3] == 0.0:
                res[i, 3:] = hpart[i, 3:]
            elif hpart[i, 3] == 0.0:
                res[i, 3:] = fpart[i, 3:]
            else:
                res[i, 3:] = quaternion_multiply(hpart[i, 3:], fpart[i, 3:])

        return res

    def decompose(self, traj: np.ndarray, hpart: np.ndarray) -> np.ndarray:
        assert traj.shape[-1] == 7, "The shape of the trajectory should be (num_samples, 7)"
        assert hpart.shape[-1] == 7, "The shape of the hpart should be (num_samples, 7)"

        ftarget = np.zeros_like(traj)

        # Calculate the position part
        ftarget[:, :3] = traj[:, :3] - hpart[:, :3]

        # Calculate the orientation part
        for i in range(len(ftarget)):
            ftarget[i, 3:] = quaternion_multiply(quaternion_conjugate(hpart[i, 3:]), traj[i, 3:])

        return ftarget

    def serialize(self, path: str) -> None:
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def deserialize(clc, path: str):
        with open(path, 'rb') as f:
            return pickle.load(f)


def test_vmp_shape_modulation():
    from vmppy.paths import VMPPath
    import matplotlib.pyplot as plt

    file = VMPPath.get_raw_motion("pickplace7d", ext="csv")
    data = np.loadtxt(file, delimiter=",")
    fig, axes = plt.subplots(1, 3, figsize=(10, 5))
    traj = TSTrajectory(data[:, 1:], data[:, 0])

    tvmp = TVMP(
        shape_modulation_config=FunctionApproximatorConfig(type="krbf", config=dict(dim=7, num_kernels=30)),
        elementary_traj_config=ElementaryTrajConfig(type="task_space_linear", dim=7),
        canonical_system_config=CanonicalSystemConfig(type="linear"),
    )
    tvmp.learn(Trajectories([traj]), num_samples=1000)
    tvmp.serialize(VMPPath.get_vmp_motion("pickplace7d", ext="pkl"))

    trajo = tvmp.rollout(1000)

    axes[0].plot(traj[:,0], 'r-', label="original-x")
    axes[0].plot(traj[:,1], 'b-', label="original-y")
    axes[0].plot(traj[:,2], 'g-', label="original-z")
    axes[0].plot(trajo[:,0], 'r--', label="tvmp-x")
    axes[0].plot(trajo[:,1], 'b--', label="tvmp-y")
    axes[0].plot(trajo[:,2], 'g--', label="tvmp-z")

    axes[1].plot(traj[:,3], 'r-', label="original-qw")
    axes[1].plot(traj[:,4], 'b-', label="original-qx")
    axes[1].plot(traj[:,5], 'g-', label="original-qy")
    axes[1].plot(traj[:,6], 'y-', label="original-qz")
    axes[1].plot(trajo[:,3], 'r--', label="tvmp-qw")
    axes[1].plot(trajo[:,4], 'b--', label="tvmp-qx")
    axes[1].plot(trajo[:,5], 'g--', label="tvmp-qy")
    axes[1].plot(trajo[:,6], 'y--', label="tvmp-qz")

    axes[2].plot(-traj[:,1], traj[:,0], 'r-')
    axes[2].plot(-trajo[:,1], trajo[:,0], 'g-')

    axes[0].legend()
    axes[1].legend()
    plt.savefig("tvmp_shape_modulation_test.png")


def test_load_vmp():
    import matplotlib.pyplot as plt
    from vmppy.paths import VMPPath
    import os

    vmp = TVMP.deserialize(VMPPath.get_vmp_motion("pickplace7d", ext="pkl"))
    orig_traj = TSTrajectory.from_file(VMPPath.get_raw_motion("pickplace7d", ext="csv"), contain_timestamps=True)
    orig_traj.normalize_timestamps(1000)
    traj = vmp.rollout(1000)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].plot(orig_traj[:,0], 'r-', label="original-x")
    axes[0].plot(orig_traj[:,1], 'b-', label="original-y")
    axes[0].plot(orig_traj[:,2], 'g-', label="original-z")
    axes[0].plot(traj[:,0], 'r-.', label="rollout-x")
    axes[0].plot(traj[:,1], 'b-.', label="rollout-y")
    axes[0].plot(traj[:,2], 'g-.', label="rollout-z")

    axes[1].plot(orig_traj[:,3], 'r-', label="original-qw")
    axes[1].plot(orig_traj[:,4], 'b-', label="original-qx")
    axes[1].plot(orig_traj[:,5], 'g-', label="original-qy")
    axes[1].plot(orig_traj[:,6], 'y-', label="original-qz")
    axes[1].plot(traj[:,3], 'r-.', label="rollout-qw")
    axes[1].plot(traj[:,4], 'b-.', label="rollout-qx")
    axes[1].plot(traj[:,5], 'g-.', label="rollout-qy")
    axes[1].plot(traj[:,6], 'y-.', label="rollout-qz")
    plt.savefig("tvmp_load_vmp_test.png")


if __name__ == "__main__":
    test_vmp_shape_modulation()
    test_load_vmp()
