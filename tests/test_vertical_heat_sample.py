import pandas as pd
import numpy as np
import pytest

from thesis_pitz_daily.vertical_heat_sample import (
    VertHeatSample,
    parse_sampled_data,
    extract_coordinates,
    extract_specific_heat,
    extract_vertical_total_heat,
)


@pytest.fixture
def sample_data():
    data = {
        "x": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        "y": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        "QdotMean": [10, 20, 30, 40, 50, 50],
    }
    return pd.DataFrame(data)


def test_extract_coordinates(sample_data):
    expected_result = np.array(
        [[1.0, 0.1], [1.0, 0.2], [1.0, 0.3], [1.0, 0.4], [1.0, 0.5]]
    )

    result = extract_coordinates(sample_data)
    assert np.array_equal(result, expected_result)


def test_extract_specific_heat(sample_data):
    expected_result = np.array([10, 20, 30, 40, 50])

    result = extract_specific_heat(sample_data)
    assert np.array_equal(result, expected_result)


def test_extract_vertical_total_heat(sample_data):
    expected_result = 150

    result = extract_vertical_total_heat(sample_data)
    assert result == expected_result


def test_parse_sampled_data(sample_data, mocker):
    file_location = "path/to/data/files"
    sampling_files = ["file1.csv", "file2.csv", "file3.csv"]

    mocker.patch("glob.glob", return_value=sampling_files)
    mocker.patch("pandas.read_csv", return_value=sample_data)

    result = parse_sampled_data(file_location)

    assert len(result) == len(sampling_files)
    assert isinstance(result[0], VertHeatSample)
