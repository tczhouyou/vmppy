from abc import abstractmethod
from enum import Enum
import numpy as np
from pydantic import BaseModel, Field



class DecayMode(Enum):
    LINEAR = "linear"
    EXPONENTIAL = "exponential"



class CanonicalSystemConfig(BaseModel):
    start: float = Field(1.0, description="Starting value of the canonical system")
    end: float = Field(0.0, description="End value of the canonical system")
    tau: float = Field(1.0, description="Scaling factor for the time dimension")
    decay_mode: DecayMode = Field(DecayMode.LINEAR, description="Type of decay: linear or exponential")


class CanonicalSystem():
    def __init__(self, start: float = 1.0, end: float = 0.0, tau: float = 1.0) -> None:
        self.start = start
        self.end = end
        self.tau = tau
        self.eps = 1e-6

    @abstractmethod
    def rollout(self, num_samples: int, **kwargs) -> np.ndarray:
        pass

    @abstractmethod
    def at(self, timestamp: float, **kwargs) -> float:
        pass
    
    def set_tau(self, tau: float) -> None:
        self.tau = tau


class LinearDecayCanonicalSystem(CanonicalSystem):
    def __init__(self):
        super().__init__(1.0, 0.0)

    def rollout(self, num_samples: int, **kwargs) -> np.ndarray:
        return np.linspace(self.start, self.end, num_samples)

    def at(self, timestamp: float, **kwargs) -> float:
        canVal = self.start + (self.end - self.start) * timestamp / self.tau        
        canVal = min(max(canVal, 0.0), 1.0)
        return canVal

class ExponentialDecayCanonicalSystem(CanonicalSystem):
    def __init__(self):
        super().__init__(1.0, 0.0)

    def rollout(self, num_samples: int, **kwargs) -> np.ndarray:
        return np.exp(np.linspace(np.log(self.start), np.log(self.end+self.eps), num_samples))

    def at(self, timestamp: float, **kwargs) -> float:
        canVal = self.start * ((self.end+self.eps) / self.start) ** timestamp
        canVal = min(max(canVal, 0.0), 1.0)
        return canVal
    

class CanonicalSystemFactory:
    @staticmethod
    def get_canonical_system(config: CanonicalSystemConfig) -> CanonicalSystem:
        if config.decay_mode == DecayMode.LINEAR:
            return LinearDecayCanonicalSystem()
        elif config.decay_mode == DecayMode.EXPONENTIAL:
            return ExponentialDecayCanonicalSystem()
        else:
            raise ValueError(f"Unsupported decay mode: {config.decay_mode}")


def test():
    pass

if __name__ == "__main__":
    test()