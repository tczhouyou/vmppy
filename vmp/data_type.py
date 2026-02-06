from typing import Optional
import numpy as np
import pickle


class ViaPoint():
    def __init__(
        self, 
        can_value: float, 
        point: np.ndarray,
        dpoint: Optional[np.ndarray] = None,
        ddpoint: Optional[np.ndarray] = None,
    ) -> None:
        self.can_value = can_value
        self.point = point
        self.dpoint = dpoint
        if self.dpoint is None:
            self.dpoint = np.zeros_like(self.point)
        
        self.ddpoint = ddpoint
        if self.ddpoint is None:
            self.ddpoint = np.zeros_like(self.point)


class ViaPose(ViaPoint):
    def __init__(self, can_value: float, point: np.ndarray) -> None:
        assert point.shape[0] == 7, "The shape of the point should be (7,)"
        super().__init__(can_value, point)

    @classmethod
    def from_position_orientation(
        cls, 
        can_value: float, 
        position: np.ndarray,
        orientation: np.ndarray
    ):
        assert position.shape[0] == 3, "The shape of the position should be (3,)"
        assert orientation.shape[0] == 4, "The shape of the orientation should be (4,)"
        pose = np.concatenate([position, orientation])
        return cls(can_value, pose)

class MVN():
    def __init__(self, center: np.ndarray, cov: np.ndarray) -> None:
        self.center = center
        self.cov = cov
    
    def sample(self) -> np.ndarray:
        return np.random.multivariate_normal(self.center, self.cov)
    
    def serialize(self, path: str) -> None:
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def deserialize(clc, path: str):
        with open(path, 'rb') as f:
            return pickle.load(f)

