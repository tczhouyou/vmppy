from copy import deepcopy
import pickle
import numpy as np
from typing import Any, Dict, List, Optional
# Allow both package import (recommended) and direct script execution.
try:
    from .trajectory import Trajectories, Trajectory
    from .compat import print_instantiation_arguments
except ImportError:  # pragma: no cover
    from vmp.trajectory import Trajectories, Trajectory
    from vmp.compat import print_instantiation_arguments

import uuid

import sys
import os
file_path = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(file_path, "../../")))
sys.path.append(os.path.abspath(os.path.join(file_path, "..")))
sys.path.append(os.path.abspath(file_path))

from vmp.data_type import ViaPoint, MVN
from vmp.canonical_system import CanonicalSystemFactory, CanonicalSystemConfig
from vmp.elementary_traj import ElementaryTrajectoryFactory, ElementaryTrajConfig
from vmp.function_approximator import FunctionApproximatorFactory, FunctionApproximatorConfig

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class VMP():
    def __init__(
        self,
        dim: int,
        name: str = "unknown",
        shape_modulation_config: FunctionApproximatorConfig = FunctionApproximatorConfig(),
        elementary_traj_config: ElementaryTrajConfig = ElementaryTrajConfig(),
        canonical_system_config: CanonicalSystemConfig = CanonicalSystemConfig(),
        **kwargs
    ) -> None:
        
        self.id = uuid.uuid4()
        if name == "unknown":
            self.name = f"VMP_{self.id}"
        else:
            self.name = name

        self.dim = dim
        config = kwargs
        config["dim"] = dim
        self.shape_modulation = FunctionApproximatorFactory.get_function_approximator(shape_modulation_config)
        self.elementary_traj = ElementaryTrajectoryFactory.get_elementary_trajectory(elementary_traj_config)
        self.canonical_system = CanonicalSystemFactory.get_canonical_system(canonical_system_config)
        self.viapoints: List[ViaPoint] = []
        self.tags = kwargs.get("tags", None)
        self.config = config

    def info(self):
        msg = f"""
=====> VMP: {self.name}, ID: {self.id}, type: {self.vmp_type}
       Start -> Goal: {self.start} -> {self.goal}
"""
        print(msg)
        print_instantiation_arguments(type(self.shape_modulation),  self.config)
        print_instantiation_arguments(type(self.elementary_traj),  self.config)
        return msg
    
    @property
    def goal(self):
        return self.get_goal()
    
    @property
    def start(self):
        return self.get_start()


    @property
    def vmp_type(self):
        return "vmp"
    
    def set_tags(self, tags: Dict[str, Any]) -> None:
        self.tags = tags
    
    def check_viapoints(self, viapoints: Optional[List[ViaPoint]] = None) -> bool:
        if viapoints is None:
            viapoints = self.viapoints

        start_exist = False
        goal_exist = False
        for viapoint in viapoints:
            if viapoint.can_value >= 1.0:
                start_exist = True
            elif viapoint.can_value <= 0.0:
                goal_exist = True

        return start_exist and goal_exist

    def get_virtual_viapoints(self, viapoints: List[ViaPoint]) -> List[ViaPoint]:
        virtual_viapoints = []
        for viapoint in viapoints:
            if viapoint.can_value >= 1.0:
                virtual_viapoints.append(viapoint)
            elif viapoint.can_value <= 0.0:
                virtual_viapoints.append(viapoint)
            else:
                fvia = self.shape_modulation(viapoint.can_value).flatten(order="F")
                hvia = viapoint.point - fvia
                virtual_viapoints.append(ViaPoint(viapoint.can_value, hvia))

        return virtual_viapoints
    
    def compose(self, fpart: np.ndarray, hpart: np.ndarray) -> np.ndarray:
        return hpart + fpart #res.reshape(-1, self.dim, order="F")
    
    def decompose(self, traj: np.ndarray, hpart: np.ndarray) -> np.ndarray:
        return traj - hpart
    
    def at(self, timestamp: float) -> np.ndarray:
        """Get the point at a specific phase value.
        
        Args:
            timestamp (float): timestamp value
        Returns:
            np.ndarray: Point at the phase value (dim,)
        """
        
        can_value = self.canonical_system.at(timestamp)
        fpart = self.shape_modulation(can_value)
        hpart = self.elementary_traj.at(can_value)
        return self.compose(fpart, hpart)

    def rollout(self, num_samples, is_deterministic=True, viapoints: Optional[List[ViaPoint]]=None) -> np.ndarray:
        """Get the points at specific phase values.
        
        Args:
            can_values (Union[np.ndarray, List[float]]): Phase values
                it can be (num_samples,) or (num_samples, 1)
            is_deterministic (bool): If True, return the mean of the distribution
            viapoints (Optional[List[ViaPoint]]): Real viapoints
        Returns:
            np.ndarray: Points at the phase values (num_samples, dim)
        """
        can_values = self.canonical_system.rollout(num_samples)
        fpart = self.shape_modulation(can_values)

        if viapoints is None:
            viapoints = self.viapoints

        if self.check_viapoints(viapoints):
            virtual_viapoints = self.get_virtual_viapoints(viapoints)
        else:
            raise ValueError("Start and goal viapoints should be set.")

        hpart = self.elementary_traj.rollout_with_viapoints(
            can_values,
            virtual_viapoints,
        )  # (num_samples, dim)

        return self.compose(fpart, hpart)

    def learn(
        self,
        trajectories: Trajectories,
        num_samples: int = 100,
        smooth_window_length_ratio: float = 0.0,
    ):
        """Learn the VMP from demonstrations.

        Args:
            trajectories (Trajectories): Demonstrations
        """
        
        trajectories.normalize_timestamps(num_samples)
        if smooth_window_length_ratio * num_samples  > 1:
            window_length = int(num_samples * smooth_window_length_ratio) # this should be an odd number
            if window_length % 2 == 0:
                window_length += 1
            trajectories.smooth(window_length=window_length)
    
        can_value = self.canonical_system.rollout(num_samples)
        can_values = [can_value] * len(trajectories)
        ftargets = []
        for traj in trajectories:
            self.learned_viapoints = [
                ViaPoint(1.0, traj.x0),
                ViaPoint(0.0, traj.xT)
            ]
            self.elementary_traj.learn(self.learned_viapoints)
            elementary_traj = self.elementary_traj.rollout(can_value) # (num_samples,)
            ftarget = self.decompose(traj.data, elementary_traj) # (num_samples, dim)
            ftargets.append(ftarget)

        can_values = np.array(can_values) # (num_demos, num_samples, )
        can_values = np.expand_dims(can_values, axis=-1) # (num_demos, num_samples, 1)
        ftargets = np.array(ftargets) # (num_demos, num_samples, dim)
        self.shape_modulation.learn(can_values, ftargets)       
        self.viapoints = deepcopy(self.learned_viapoints)

    def insert_viapoint(self, viapoint: ViaPoint) -> None:
        """Insert a virtual viapoint.
        
        Args:
            viapoint (ViaPoint): Virtual viapoint
        """
        for vp in self.viapoints:
            if vp.can_value == viapoint.can_value:
                self.viapoints.remove(vp)

        self.viapoints.append(viapoint)
        virtual_viapoints = self.get_virtual_viapoints(self.viapoints)
        self.elementary_traj.learn(virtual_viapoints)

    def insert_viapoints(self, viapoints: List[ViaPoint]) -> None:
        """Insert a virtual viapoint.
        
        Args:
            viapoints (List[ViaPoint]): Virtual viapoints
        """
        if len(viapoints) == 0:
            return

        self.viapoints.extend(viapoints)
        virtual_viapoints = self.get_virtual_viapoints(self.viapoints)
        self.elementary_traj.learn(virtual_viapoints)

    def set_tau(self, tau: float) -> None:
        self.canonical_system.set_tau(tau)

    def get_tau(self) -> float:
        return self.canonical_system.tau

    def reset(self) -> None:
        self.weights = deepcopy(self.learned_weights)
        self.viapoints = deepcopy(self.learned_viapoints)

    def save_learned_weights(self, path: str) -> None:
        self.learned_weights.serialize(path)

    def load_learned_weights(self, path: str) -> None:
        self.learned_weights = MVN.deserialize(path)
        self.weights = deepcopy(self.learned_weights)

    def save_viapoints(self, path: str) -> None:
        with open(path, 'wb') as f:
            pickle.dump(self.viapoints, f)

    def load_viapoints(self, path: str) -> None:
        with open(path, 'rb') as f:
            self.viapoints = pickle.load(f)

    def serialize(self, path: str) -> None:
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def deserialize(clc, path: str):
        with open(path, 'rb') as f:
            return pickle.load(f)
        
    def set_start(self, start: np.ndarray) -> None:
        for viapoint in self.viapoints:
            if viapoint.can_value == 1.0:
                self.viapoints.remove(viapoint)
        
        self.insert_viapoint(ViaPoint(1.0, start))

    def set_goal(self, goal: np.ndarray) -> None:
        if goal.shape[-1] != self.dim:
            return
        
        for viapoint in self.viapoints:
            if viapoint.can_value == 0.0:
                self.viapoints.remove(viapoint)

        self.insert_viapoint(ViaPoint(0.0, goal))

    def clean_viapoints(self) -> None:
        for viapoint in self.viapoints:
            if viapoint.can_value != 1.0 and viapoint.can_value != 0.0:
                self.viapoints.remove(viapoint)

        virtual_viapoints = self.get_virtual_viapoints(self.viapoints)
        self.elementary_traj.learn(virtual_viapoints)

    def learn_from_files(
        self, 
        files: List[str], 
        compress_ratio: float = 1.0, 
        num_samples: Optional[int] = None,
        smooth_window_length_ratio: float = 0.0,
        extend_ends: int = 0
    ) -> None:
        trajectory = Trajectory.from_file(files[0])
        if num_samples is None:
            if compress_ratio >= 1.0:
                num_samples = int(trajectory.data.shape[0] / compress_ratio)
            else:
                num_samples = 20
        trajectories = Trajectories.from_files(files)
        for traj in trajectories:
            traj.extend_ends(extend_ends)

        self.learn(trajectories, num_samples, smooth_window_length_ratio)

    def learn_from_file(
        self, 
        file: str, 
        compress_ratio: float = 1.0, 
        num_samples: Optional[int] = None
    ) -> None:
        self.learn_from_files([file], compress_ratio, num_samples)

    
    def get_goal(self) -> np.ndarray:
        for viapoint in self.viapoints:
            if viapoint.can_value == 0.0:
                return viapoint.point
            
    def get_start(self) -> np.ndarray:
        for viapoint in self.viapoints:
            if viapoint.can_value == 1.0:
                return viapoint.point
    

if __name__ == "__main__":
    timestamps = np.linspace(0, 1, 1000)
    noise_scale = 0.01

    data = np.zeros((1000, 2))
    data[:,0] = np.sin(timestamps * 2 * np.pi) + timestamps + noise_scale * (np.random.random(timestamps.shape) - 0.5)
    data[:,1] = np.cos(timestamps * 2 * np.pi) + timestamps + noise_scale * (np.random.random(timestamps.shape) - 0.5)
    traj0 = Trajectory(data, timestamps)

    data = np.zeros((1000, 2))
    data[:,0] = np.sin(timestamps * 2 * np.pi) + timestamps + noise_scale * (np.random.random(timestamps.shape) - 0.5)
    data[:,1] = np.cos(timestamps * 2 * np.pi) + timestamps + noise_scale * (np.random.random(timestamps.shape) - 0.5)
    traj1 = Trajectory(data, timestamps)

    trajectories = Trajectories([traj0])#, traj1])
    
    # vmp = VMP(
    #     dim=2, 
    #     shape_modulation_type='fbfcnn', 
    #     train_epoch=200, 
    #     hidden_sizes=[64,64], 
    #     device="cuda"
    # )

    vmp = VMP(dim=2, shape_modulation_type="hash")

    vmp.learn(trajectories, num_samples=100)
    vmp.insert_viapoint(
        ViaPoint(
            0.5,
            np.array([0.0, 0.0])
        )
    )

    vmp.serialize("vmp_learned.pkl")

    vmp1 = VMP.deserialize("vmp_learned.pkl")
    test_traj = vmp1.rollout(100)
    import matplotlib.pyplot as plt
    traj0.plot()
    traj1.plot()
    plt.plot(traj0.timestamps, test_traj)
    plt.plot(0.5, 0.0, "ro")
    plt.show()
