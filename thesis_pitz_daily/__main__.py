from thesis_pitz_daily.vertical_heat_sample import parse_sampled_data
from thesis_pitz_daily.sample_heat_gaussian import (
    generate_normal_fits,
    save_normal_fit_plot,
)
from thesis_pitz_daily.dependancies import plot_relative_error

# TODO: add logging


def main() -> None:
    center_path = "data/CenterQdotMean"
    linear_path = "data/QdotMean"

    heat_sources = parse_sampled_data(linear_path, center_path)
    normal_fits = generate_normal_fits(heat_sources)
    print("hello")
    # save_normal_fit_plot(normal_fits)
    plot_relative_error(normal_fits)


if __name__ == "__main__":
    main()
