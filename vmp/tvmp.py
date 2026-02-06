import os
import numpy as np
np.bool = np.bool_

from typing import List
import pickle

# Allow both package import (recommended) and direct script execution.
try:
    from .transformations import quaternion_multiply, quaternion_conjugate
    from .trajectory import Trajectories
    from .trajectory import TSTrajectory
except ImportError:  # pragma: no cover
    from vmp.transformations import quaternion_multiply, quaternion_conjugate
    from vmp.trajectory import Trajectories, TSTrajectory


import sys
file_path = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(file_path, "../../")))
sys.path.append(os.path.abspath(os.path.join(file_path, "..")))
sys.path.append(os.path.abspath(file_path))


from vmp.data_type import ViaPose
from vmp.vmp import VMP
from vmp.canonical_system import CanonicalSystemConfig
from vmp.elementary_traj import ElementaryTrajConfig
from vmp.function_approximator import FunctionApproximatorConfig


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
    import matplotlib.pyplot as plt
    import copy
    # file ="/home/unix_ai/unix_manipulation_project/data/motion_library/hand_writing/clean_pen.npy"
    file = "/home/unix_ai/unix_manipulation_project/data/motion_library/hand_writing/raise_arm_final.npy"
    data = np.load(file)[:,7:]
    fig, axes = plt.subplots(1, 3, figsize=(10, 5))

    traj = TSTrajectory(data)
    traj2 = copy.deepcopy(traj)
    # traj.shrink(0.5, dims=[0,1])
    tvmp = TVMP(shape_modulation_type="krbf", num_kernels=50)
    tvmp.learn(Trajectories([traj]), num_samples=1000)
    
    # tvmp2 = TVMP(shape_modulation_type="fbfcnn", train_epochs=100, stop_threshold=1e-6, hidden_sizes=[256, 256, 256], batch_size=128)
    # tvmp2 = TVMP(
    #     shape_modulation_type="fbgru", 
    #     train_epochs=200, 
    #     stop_threshold=1e-6, 
    #     hidden_size=256, 
    #     num_layers=2, 
    #     batch_size=256,
    #     subseq_len=200,
    # )
    tvmp2 = TVMP(shape_modulation_type="hash", table_size=10000)
    tvmp2.learn(Trajectories([traj2]), num_samples=10000)

    traja = tvmp.rollout(1000)
    trajb = tvmp2.rollout(1000)

    axes[0].plot(traj[:,0], 'r-')
    axes[0].plot(traj[:,1], 'b-')
    axes[0].plot(traj[:,2], 'g-')
    axes[0].plot(traja[:,0], 'r-.')
    axes[0].plot(traja[:,1], 'b-.')
    axes[0].plot(traja[:,2], 'g-.')
    axes[0].plot(trajb[:,0], 'r--')
    axes[0].plot(trajb[:,1], 'b--')
    axes[0].plot(trajb[:,2], 'g--')

    axes[1].plot(traj[:,3], 'r-')
    axes[1].plot(traj[:,4], 'b-')
    axes[1].plot(traj[:,5], 'g-')
    axes[1].plot(traj[:,6], 'y-')
    axes[1].plot(traja[:,3], 'r-.')
    axes[1].plot(traja[:,4], 'b-.')
    axes[1].plot(traja[:,5], 'g-.')
    axes[1].plot(traja[:,6], 'y-.')
    axes[1].plot(trajb[:,3], 'r--')
    axes[1].plot(trajb[:,4], 'b--')
    axes[1].plot(trajb[:,5], 'g--')
    axes[1].plot(trajb[:,6], 'y--')


    axes[2].plot(-traj[:,1], traj[:,0], 'r-')
    axes[2].plot(-traja[:,1], traja[:,0], 'b-')
    axes[2].plot(-trajb[:,1], trajb[:,0], 'g-')
    # axes[2].plot(-traj2[:,1], traj2[:,0], 'g.')
    plt.show()


def test_load_vmp():
    import matplotlib.pyplot as plt
    from .paths import VMPPath
    import os

    vmp = TVMP.deserialize(os.path.join(VMPPath.VMP_MOTION_PATH, "raise_right_arm.pkl"))
    orig_traj = TSTrajectory.from_file("/home/unix_ai/unix_manipulation_project/data/motion_library/common/raise_right_arm.npy")
    orig_traj.normalize_timestamps(1000)
    traj = vmp.rollout(1000)
    print(vmp.viapoints)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].plot(orig_traj[:,0], 'r-')
    axes[0].plot(orig_traj[:,1], 'b-')
    axes[0].plot(orig_traj[:,2], 'g-')
    axes[0].plot(traj[:,0], 'r-.')
    axes[0].plot(traj[:,1], 'b-.')
    axes[0].plot(traj[:,2], 'g-.')

    axes[1].plot(orig_traj[:,3], 'r-')
    axes[1].plot(orig_traj[:,4], 'b-')
    axes[1].plot(orig_traj[:,5], 'g-')
    axes[1].plot(orig_traj[:,6], 'y-')
    axes[1].plot(traj[:,3], 'r-.')
    axes[1].plot(traj[:,4], 'b-.')
    axes[1].plot(traj[:,5], 'g-.')
    axes[1].plot(traj[:,6], 'y-')
    plt.show()


if __name__ == "__main__":
    # test()

    test_load_vmp()
    