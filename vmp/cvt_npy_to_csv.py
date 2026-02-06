import numpy as np
from argparse import ArgumentParser


def cvt_npy_to_csv(npy_file, csv_file):
    data = np.load(npy_file)
    np.savetxt(csv_file, data, delimiter=",")
    
    
if __name__ == "__main__":
    import os
    parser = ArgumentParser()
    parser.add_argument("--npy_folder", type=str, required=True)
    args = parser.parse_args()
    
    npy_folder = args.npy_folder
    npy_files = [f for f in os.listdir(npy_folder) if f.endswith(".npy")]
    for npy_file in npy_files:
        csv_file = npy_file.replace(".npy", ".csv")
        cvt_npy_to_csv(os.path.join(npy_folder, npy_file), os.path.join(npy_folder, csv_file))
        
    