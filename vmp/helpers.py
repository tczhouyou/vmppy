from typing import Any, Dict, Optional, Literal, List
import numpy as np
from pydantic import BaseModel, Field, model_validator
import os

from .data_type import ViaPoint
from .vmp import VMP
from .tvmp import TVMP
from .canonical_system import CanonicalSystemConfig
from .elementary_traj import ElementaryTrajConfig
from .function_approximator import FunctionApproximatorConfig, KRBFConfig
from .paths import VMPPath


class ViaPointConfig(BaseModel):
    canval: float
    viapoint: List[float]

    def to_viapoint(self) -> ViaPoint:
        return ViaPoint(can_value=self.canval, point=np.array(self.viapoint))


class VMPLearnConfig(BaseModel):
    compress_ratio: float = 1.0
    num_samples: Optional[int] = None
    smooth_window_length_ratio: float = 0.0
    extend_ends: int = 0


class VMPRunTimeConfig(BaseModel):
    goal: Optional[List[float]] = None
    viapoints: Optional[List[ViaPointConfig]] = None
    duration: Optional[float] = None


class VMPConfig(BaseModel):
    name: str = "unknown"
    type: Literal["vmp", "tvmp"]
    dim: int = 7
    vmp_file: Optional[str] = None
    trajectory_file: Optional[str] = None
    canonical_system_config: CanonicalSystemConfig = Field(default_factory=CanonicalSystemConfig)
    elementary_trajectory_config: Optional[ElementaryTrajConfig] = None
    shape_modulation_config: Optional[FunctionApproximatorConfig] = None
    learn_config: VMPLearnConfig = Field(default_factory=VMPLearnConfig)
    runtime_config: Optional[VMPRunTimeConfig] = None

    @model_validator(mode="before")
    def check_file_paths(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        if values.get('vmp_file') is None and values.get('trajectory_file') is None:
            raise ValueError("Either 'vmp_file' or 'trajectory_file' must be provided.")
        return values
        
    def model_post_init(self, __context: Any) -> None:
        if self.elementary_trajectory_config is None:
            self.elementary_trajectory_config = ElementaryTrajConfig(
                type="task_space_linear" if self.type == "tvmp" else "linear", dim=self.dim
            )

        if self.shape_modulation_config is None:
            self.shape_modulation_config = FunctionApproximatorConfig(
                type="krbf", config=KRBFConfig(dim=self.dim, num_kernels=20)
            )

    @model_validator(mode="after")
    def check_vmpconfig(cls, values: "VMPConfig") -> "VMPConfig":
        if values.type == "tvmp":
            values.elementary_trajectory_config.dim = 7
            values.shape_modulation_config.config.dim = 7
        
        return values

def instantiate_vmp_from_cfg(cfg: VMPConfig) -> VMP:
    """Instantiate VMP based on the VMPConfig."""
    vmp_type = cfg.type
    if cfg.vmp_file:
        vmp_file = VMPPath.get_vmp_motion(cfg.vmp_file)
        vmp = VMP.deserialize(vmp_file) if vmp_type == "vmp" else TVMP.deserialize(vmp_file)
    elif cfg.trajectory_file:
        # Instantiate based on type
        vmp = (
            VMP(
                name=cfg.name,
                dim=cfg.dim, 
                shape_modulation_config=cfg.shape_modulation_config,
                elementary_traj_config=cfg.elementary_trajectory_config,
                canonical_system_config=cfg.canonical_system_config
            )
            if vmp_type == "vmp" else
            TVMP(
                name=cfg.name,
                shape_modulation_config=cfg.shape_modulation_config,
                elementary_traj_config=cfg.elementary_trajectory_config,
                canonical_system_config=cfg.canonical_system_config
            )
        )

        trajectory_file = os.path.join(VMPPath.RAW_MOTION_PATH, cfg.trajectory_file)
        vmp.learn_from_files([trajectory_file], **cfg.learn_config.model_dump())
    else:
        raise ValueError("cfg must contain either 'vmp_file' or 'trajectory_file'.")        

    vmp.info()

    if cfg.runtime_config:
        if cfg.runtime_config.duration:
            vmp.set_tau(cfg.runtime_config.duration)
        if cfg.runtime_config.viapoints:
            viapoints = [ViaPoint(can_value=vp_cfg.canval, point=np.array(vp_cfg.viapoint))
                         for vp_cfg in cfg.runtime_config.viapoints]
            vmp.insert_viapoints(viapoints)
        if cfg.runtime_config.goal:
            vmp.set_goal(np.array(cfg.runtime_config.goal))

    return vmp
