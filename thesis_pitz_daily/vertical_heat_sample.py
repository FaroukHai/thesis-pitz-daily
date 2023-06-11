"""
The module for vertical heat source sample line.
"""
import glob
from dataclasses import dataclass
import pandas as pd
import numpy as np
from tqdm import tqdm


@dataclass
class VertHeatSample:
    """
    This is the sampled data of a vertical sample line of a PitzDaily CFD simulation.
    It requires the data input and generation to be done by another function.
    """

    x_position: float
    y_position: np.ndarray
    coordinates: np.ndarray
    specific_heats: np.ndarray
    heat: int  # Linear if false.
    spacing_norm = 0.000204  # Vertical Distance between datapoints

    def __len__(self):
        """
        The number of samples.
        """
        return len(self.coordinates)


def parse_sampled_data(linear_location: str, cell_center_location: str) -> np.ndarray:
    """
    The sampled vertical lines need to be instentiated from the data.
    Data from the linear and cell centered sampling is used.
    """
    linear_files = glob.glob(linear_location + "/*.csv")
    cell_center_files = glob.glob(cell_center_location + "/*.csv")
    heat_source_objects = []

    for linear_file, cell_center_file in tqdm(zip(linear_files, cell_center_files[:])):
        linear_df = pd.read_csv(linear_file)
        cell_center_df = pd.read_csv(cell_center_file)
        heat_sample = VertHeatSample(
            linear_df["x"][0],
            linear_df["y"].values,
            extract_coordinates(linear_df),
            extract_specific_heat(linear_df),
            extract_vertical_total_heat(cell_center_df),
        )
        heat_source_objects.append(heat_sample)
    return np.array(heat_source_objects)


def extract_coordinates(file_data: pd.DataFrame) -> np.ndarray:
    """
    Generates a numpy array of coordinate arrays of the sampled positions.
    """
    x_position = file_data["x"][0]
    y_positions = file_data["y"].values
    return np.column_stack(
        (
            np.full_like(y_positions, x_position),
            y_positions,
        )
    )


def extract_specific_heat(file_data: pd.DataFrame) -> np.ndarray:
    """
    The specific heat values of a sample line in the CFD simulation.
    """
    return file_data["QdotMean"].values


def extract_vertical_total_heat(file_data: pd.DataFrame) -> float:
    """
    Gets the total vertical volume-normal heat extracted.
    """
    cell_center_heat = np.unique(file_data["QdotMean"].values)
    return np.sum(cell_center_heat)
