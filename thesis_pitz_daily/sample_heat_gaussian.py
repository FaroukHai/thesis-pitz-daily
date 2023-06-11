"""
This module is for generating the gaussian distributions from the heat source
sampled data.
"""

from dataclasses import dataclass

from tqdm import tqdm
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
import matplotlib.pyplot as plt

from thesis_pitz_daily.vertical_heat_sample import VertHeatSample


@dataclass
class SampleHeatGaussian:
    """
    A dataclass to hold the gaussian data and allow an output plot.
    """

    mean: float
    std_dev: float
    original_sample: VertHeatSample
    relative_error: float
    sample_y_norm: np.ndarray
    sample_heat_norm: np.ndarray

    def boiler_gauss_plot(self) -> None:
        """
        Boilerplate code to plot the gaussian.
        """
        plt.figure()
        plt.plot(
            self.original_sample.y_position,
            self.sample_heat_norm,
            "ok",
            label="Heat Distribution",
        )
        plt.plot(
            self.original_sample.y_position,
            norm(self.std_dev, self.mean).pdf(self.sample_y_norm),
            "--r",
            label=f"Curve fit - {str(float(round(self.relative_error * 100, 3)))} %",
        )
        plt.title(
            f"Heat distribution fit at x = {str(self.original_sample.x_position)} mm"
        )
        plt.xlabel("y-position in mm")
        plt.ylabel("Normalized Heat intensity")
        plt.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")
        plt.grid()

    def save_plot_gauss(self, save_output="output/plots/norm_dist") -> None:
        """
        Plot the gaussian and save the plot.
        """
        self.boiler_gauss_plot()
        plt.savefig(
            f"{save_output}/{self.original_sample.x_position}_norm_heat_sourceplot.svg",
            format="svg",
        )
        plt.close()

    def show_plot(self) -> None:
        """
        Plots the gaussian and shows it.
        """
        self.boiler_gauss_plot()
        plt.show()


class GeneratedSampleGaussian:
    """
    A class that generates the gaussian from the sampel inputs.
    """

    def __init__(self, original_sample: VertHeatSample):
        self.original_sample = original_sample

    def norm_y_position(self) -> np.ndarray:
        """
        Normalizes the y values based on the sampling.
        """
        return self.original_sample.y_position / self.original_sample.spacing_norm

    def norm_specific_heat_samples(self) -> np.ndarray:
        """
        Normalizes the heat samples, and where the value may be negative, it is set to 0.
        Only unieue values are initially taken as to not oversample.
        """

        cleaned_samples = np.where(
            self.original_sample.specific_heats < 0,
            0.0,
            self.original_sample.specific_heats,
        )
        return cleaned_samples / sum(cleaned_samples)

    def y_sample_length_delta(self) -> np.ndarray:
        """
        Calculatest the space between all y samples.
        """
        difference_array = (
            self.original_sample.y_position[:, -1]
            - self.original_sample.y_position[1, :]
        )
        return np.sum(np.abs(difference_array))

    def _initialization(self) -> tuple[float]:
        """
        Initialiyes mu and sigma for optimization.
        """
        mu = np.sum(
            self.norm_y_position() * self.norm_specific_heat_samples()
        ) / np.sum(self.norm_specific_heat_samples())
        sigma = np.sqrt(
            np.abs(
                np.sum(
                    (self.norm_y_position() - mu) ** 2
                    * self.norm_specific_heat_samples()
                )
                / np.sum(self.norm_specific_heat_samples())
            )
        )
        return mu, sigma

    def fit_to_gaussian(self) -> tuple:
        """
        The optimization process to fit a normal distribution to the data takes
        place in this function.
        """
        init = self._initialization()
        y_norm = self.norm_y_position()
        heat_norm = self.norm_specific_heat_samples()

        def _obj_func(x: np.ndarray, y_norm, heat_norm):
            # x = [mean, std]
            norm_dist = norm(x[1], x[0]).pdf(y_norm)
            return (
                100 * np.linalg.norm(norm_dist - heat_norm) / np.linalg.norm(heat_norm)
            )

        result = minimize(
            fun=_obj_func,
            x0=np.array([init[1], init[0]]),
            args=(y_norm, heat_norm),
        )
        return result.x

    def relative_error(self, gaussian_fit) -> float:
        """
        Calculating the relative error achieved.
        """
        fit_gaussian = norm(gaussian_fit[1], gaussian_fit[0]).pdf(
            self.norm_y_position()
        )

        return np.linalg.norm(
            fit_gaussian - self.norm_specific_heat_samples()
        ) / np.linalg.norm(self.norm_specific_heat_samples())

    def generate_gaussian(self) -> SampleHeatGaussian:
        """
        Generate the gaussian dataclass that will be used for visualization.
        """
        gaussian_fit = self.fit_to_gaussian()
        return SampleHeatGaussian(
            gaussian_fit[0],
            gaussian_fit[1],
            self.original_sample,
            self.relative_error(gaussian_fit),
            self.norm_y_position(),
            self.norm_specific_heat_samples(),
        )


def generate_normal_fits(
    heat_sources: np.ndarray,
) -> np.ndarray:
    """
    Generates all the normal distribution to then recreate the gaussian.
    """
    gaussian_generators = [
        GeneratedSampleGaussian(sampled_heat_source)
        for sampled_heat_source in tqdm(heat_sources)
    ]
    return [gaussian.generate_gaussian() for gaussian in tqdm(gaussian_generators)]


def save_normal_fit_plot(fitted_hs: np.ndarray) -> None:
    """
    Generates and saves all the plots.
    """
    for heat_source in tqdm(fitted_hs):
        heat_source.save_plot_gauss()
