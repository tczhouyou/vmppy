from pathlib import Path

repls = {
    'from robot_utils.trajectory import Trajectories, Trajectory': 'from .trajectory import Trajectories, Trajectory',
    'from robot_utils.trajectory import Trajectories': 'from .trajectory import Trajectories',
    'from robot_utils.trajectory import Trajectory': 'from .trajectory import Trajectory',
    'from robot_utils.trajectory import TSTrajectory': 'from .trajectory import TSTrajectory',
    'from robot_utils.math.transformations import quaternion_slerp': 'from .transformations import quaternion_slerp',
    'from robot_utils.math.transformations import quaternion_multiply, quaternion_conjugate': 'from .transformations import quaternion_multiply, quaternion_conjugate',
    'from robot_utils.files.const_path import VMPPath': 'from .paths import VMPPath',
    'from robot_utils.files import VMPPath': 'from .paths import VMPPath',
    'from pyutils.functions_tools import print_instantiation_arguments': 'from .compat import print_instantiation_arguments',
}

for p in Path('vmp').glob('*.py'):
    txt = p.read_text()
    for a, b in repls.items():
        txt = txt.replace(a, b)
    p.write_text(txt)

print('done')
