import numpy as np
from argparse import ArgumentParser
from vmp.vmp import VMP
from vmp.tvmp import TVMP
from matplotlib import pyplot as plt

parser = ArgumentParser("VMP Learner")
parser.add_argument("--pkl", type=str, default="learned_vmp_cfg")
parser.add_argument("--type", type=str, default="tvmp")
parser.add_argument("--start", type=str, default="11, 22, 33, 44, 55, 66, 77")
parser.add_argument("--goal", type=str, default="11, 22, 33, 44, 55, 66, 77")
parsed_args = parser.parse_args()
pkl_file = parsed_args.pkl

if parsed_args.type == "vmp":
    vmp: VMP = VMP.deserialize(pkl_file)
else:
    vmp: TVMP = TVMP.deserialize(pkl_file)


start = np.array([float(x.strip()) for x in parsed_args.start.split(",")])
goal = np.array([float(x.strip()) for x in parsed_args.goal.split(",")])

vmp.set_start(start)
vmp.set_goal(goal)

traj = vmp.rollout(num_samples=100)

plt.plot(traj[:, :3])
plt.savefig("vmp_rollout.png")
