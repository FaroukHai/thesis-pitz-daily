"""
List of functions that could be used in many cases.
"""
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from thesis_pitz_daily.sample_heat_gaussian import SampleHeatGaussian


# Not used, can be delted.
def normal_distribution(x_val: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """
    Normal distribution (gaussian), function.
    """
    return np.exp(-((x_val - mu) ** 2) / (2 * sigma**2))


def plot_relative_error(
    fitted_gaussians: list[SampleHeatGaussian], save_output="output/plots"
) -> None:
    pos_error = np.array(
        [
            [fitted_gaussian.original_sample.x_position, fitted_gaussian.relative_error]
            for fitted_gaussian in tqdm(fitted_gaussians)
        ]
    )

    sorted_indices = np.argsort(pos_error[:, 0])
    pos_error = pos_error[sorted_indices]

    plt.figure()
    plt.plot(pos_error[:, 0].T, pos_error[:, 1].T * 100, "--k")

    plt.title("Gauss fit error in %")
    plt.xlabel("x-position in m")
    plt.ylabel("Gauss fit error")
    plt.grid()
    plt.show()

    plt.savefig(
        f"{save_output}/relative_error_in_flow_direction.svg",
        format="svg",
    )
