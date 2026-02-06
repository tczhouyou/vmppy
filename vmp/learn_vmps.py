from .paths import VMPPath
import hydra
from argparse import ArgumentParser
import logging
from vmp.helpers import instantiate_vmp_from_cfg, VMPConfig


from vmp.vmp import VMP
import matplotlib.pyplot as plt
import numpy as np


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def learn_vmps():
    """Learn VMP from the trajectory file and store the VMP object to the vmp_file.
    
    The config should look like the following:
    type: tvmp
    trajectory_file: "xxx"
    canonical_system_config:
      type: linear
    elementary_trajectory_config:
      type: task_space_linear
    shape_modulation_config:
      type: krbf
      num_kernels: 20
    vmp_file_to_store: "xxx"
    """
    import sys
    import os
    parser = ArgumentParser("VMP Learner")
    parser.add_argument("--main-config", type=str, default="learned_vmp_cfg")
    
    parsed_args = parser.parse_args(sys.argv[1:])

    with hydra.initialize(config_path="configs"):
        config = hydra.compose(config_name=parsed_args.main_config)
        
    for vmp_cfg in config.vmps:
        vmp_file_to_store = vmp_cfg["vmp_file_to_store"]
        vmp_cfg = VMPConfig(**vmp_cfg)
        
        vmp = instantiate_vmp_from_cfg(vmp_cfg)
        vmp_file = os.path.join(VMPPath.VMP_MOTION_PATH, vmp_file_to_store)
        vmp.serialize(vmp_file)
        
        # if vmp_file_to_store.startswith("p2p_right_arm"):
        #   cvmp = VMP.deserialize(vmp_file)
        #   cvmp.set_goal(np.array([0.6, 0.6, 0.6, 1.0, 0.0, 0.0, 0.0]))
        #   traj = cvmp.rollout(100)
        #   plt.plot(traj[:,:3], 'r-')
        #   plt.show()
        
        print(f"VMP learned from {vmp_cfg.trajectory_file} is stored in {vmp_file_to_store}")


if __name__ == "__main__":
    learn_vmps()