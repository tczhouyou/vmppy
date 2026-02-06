import os
import re
import pandas as pd
import numpy as np


class FolderHandler():
    @staticmethod
    def sorted_files(folder_path: str) -> list:
        return sorted([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])

    @staticmethod
    def sorted_files_with_numbers(folder_path: str) -> list:
        files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

        # Regular expression to match the filenames
        pattern = re.compile(r'trajectories-(\d+)')

        # Extract the number and sort the files
        sorted_files = sorted(
            files,
            key=lambda f: int(pattern.search(f).group(1)) if pattern.search(f) else float('inf')
        )

        return sorted_files


class CSVReader():
    def __init__(self, csv_path: str) -> None:
        self.csv_path = csv_path
        self.content = self.read_csv()

    def check_header(self) -> bool:
        first_row = pd.read_csv(self.csv_path, nrows=1)
        first_data = pd.read_csv(self.csv_path, nrows=1, header=None)

        if first_row.columns.equals(first_data.iloc[0]):
            return False
        else:
            return True

    def read_csv(self) -> pd.DataFrame:
        if self.check_header():
            return pd.read_csv(self.csv_path)
        else:
            return pd.read_csv(self.csv_path, header=None)

    @property
    def pd_data(self) -> pd.DataFrame:
        return self.content

    @property
    def np_data(self) -> np.ndarray:
        return self.content.to_numpy()

    @property
    def columns(self) -> list:
        return self.content.columns.tolist()

