#!/usr/bin/env python3
from pathlib import Path

import argh
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from itertools import combinations
from matplotlib import rc

# from .split_histogram import *
import split_histograms as sh

""" Plot the properties of the granules of various treaments. """

def main(input_file: Path, out_dir: Path = "/tmp/", comp_file: Path = None, bins=8, density=False, latex=True, img_path_filter:str = None):
    """Create the plots."""

    if latex:
        rc("text", usetex=True)
        rc("text.latex", preamble=r"\usepackage{siunitx, physics}")
        rc("font", family=r"serif", size=10)
        rc("mathtext", fontset="cm")

    input_file = Path(input_file)
    output_dir = Path(out_dir)
    if comp_file:
        comp_file = Path(comp_file)
        granule_data = sh.read_data(input_file, comp_file)
    else :
        granule_data = sh.read_data_new(input_file)

    print("\n===================\n")
    if img_path_filter is not None:
        for index, row in granule_data.iterrows():
            granule_data['image_path'][index] = str(granule_data['image_path'][index])
        granule_data = granule_data[granule_data['image_path'].str.endswith(img_path_filter)==True]
    # Add progressive filters to work out how many granules passed each filtering step.
    filters = ["sigma > 1e-10 and pass_rate > 0.6 and fitting_error < 0.5 and fitting_diff  > 0.03",
               None,"sigma > 1e-10 and pass_rate > 0.6",
               "sigma > 1e-10 and pass_rate > 0.6 and fitting_error < 0.5"
    ]
    for filter in filters:
        if filter is None:
            granule_filter_query_results = granule_data
        else:
            granule_filter_query_results = granule_data.query(filter, inplace=False)
        for label, chunk in granule_filter_query_results.groupby("treatment"):
            n_granules = len(chunk)
            if label == 'unknown':
                print(f"{n_granules} granules when using filter: '{filter}'")
            else:
                print(f"for {label}, {n_granules} granules when using filter: '{filter}'")
    print("\n===================\n")

    # Ensure the granules are fully filtered and plot the 1D and 2D histograms.

    granule_data.query(
        "sigma > 1e-10 and pass_rate > 0.6 and fitting_error < 0.5 and fitting_diff  > 0.03"# and granule_id < 150",# and mean_radius > 1.2",# and mean_radius > 0.4 and mean_radius < 0.6",
        inplace=True,
    )
    granule_data["sigma_micro"] = granule_data["sigma"] * 1e6

    # Pair plot does not work well with latex for some reason
    if not latex:
        pair_plot(granule_data, out_dir)

    hist_plots(granule_data, out_dir, density=density)
    draft = False
    micro_units = r"μN/m" if draft else r"\si{\micro N/m}"

    specs = [("times","Times(s)",False),
             ("sigma", "Interfacial Tension (N/m)",True),
             ("kappa_scale", "Bending Rigidity ($k_{\mathrm{B}}T$)",True),
             ("sigma_err", "Surface Tension Error (N/m)",True),
             ("kappa_scale_err", "Bending Rigidity Error($k_{\mathrm{B}}T$)",True),
             ("fitting_error", "Fitting Error",False),
             ("q_2_mag", "$|C_2|^2$",False),
             ("mean_radius","Mean Radius",False),
             ("pass_rate","Pass Rate",False),
             ("mean_intensity","Intensity",False)]

    for x, y in combinations(specs, 2):
        x_name, x_label, x_log = x
        y_name, y_label, y_log = y
        sh.histogram2D(
            y_name,
            y_label,
            x_name,
            x_label,
            granule_data,
            out_dir=out_dir,
            log_scaleX=x_log,
            log_scaleY=y_log,
            legend = False
        )

def hist_plots(granule_data: pd.DataFrame, out_dir: Path, density=False):

    sigma_bins = np.logspace(-9.5, -4, 60)
    sh.overlap_hist(
        "sigma",
        "treatment",
        "Surface Tension (N/m)",
        split_label=None,
        bins=sigma_bins,
        granule_data=granule_data,
        out_dir=out_dir,
        density=density,
        plot_errors="sigma_err",
        benchling_format=True
    )

    # sigma_bins = np.logspace(-9.5, -4, 60)
    # sh.overlap_hist(
    #     "sigma_st",
    #     "treatment",
    #     "Surface Tension (Only) (N/m)",
    #     split_label=None,
    #     bins=sigma_bins,
    #     granule_data=granule_data,
    #     out_dir=out_dir,
    #     density=density,
    #     plot_errors="sigma_errST",
    # )

    kappa_bins = np.logspace(-3, 3, 72)#(-3, 2, 60)
    sh.overlap_hist(
        "kappa_scale",
        "treatment",
        "Bending Rigidity ($k_{\mathrm{B}}T$)",
        split_label=None,
        bins=kappa_bins,
        granule_data=granule_data,
        out_dir=out_dir,
        density=density,
        plot_errors="kappa_scale_err",
        benchling_format=True
    )

    radius_bins = 60
    draft = False
    micro_units = "μm" if draft else r"\si{\micro m}"
    radius_label = f"Mean Radius ({micro_units})"
    sh.overlap_hist(
        "mean_radius",
        "treatment",
        radius_label,
        "Treatment",
        bins=radius_bins,
        granule_data=granule_data,
        out_dir=out_dir,
        density=density,
        benchling_format=True
    )

    error_bins = 60
    error_label = f"Goodness of fit"
    sh.overlap_hist(
        "fitting_error",
        "treatment",
        error_label,
        "Treatment",
        bins=error_bins,
        granule_data=granule_data,
        out_dir=out_dir,
        density=density,
        benchling_format=True
    )

    error_bins = 60
    error_label = f"Goodness of fit difference"
    sh.overlap_hist(
        "fitting_diff",
        "treatment",
        error_label,
        "Treatment",
        bins=error_bins,
        granule_data=granule_data,
        out_dir=out_dir,
        density=density,
        is_log = False,
        benchling_format=True
    )

    intensity_bins = 60
    intensity_label = "Intensity"
    sh.overlap_hist(
        "mean_intensity",
        "treatment",
        intensity_label,
        "Treatment",
        bins=intensity_bins,
        granule_data=granule_data,
        out_dir=out_dir,
        density=density,
        benchling_format=True
    )

    q2_bins = np.logspace(-5, 1, 60)
    q2_label = "q2"
    sh.overlap_hist(
        "q_2_mag",
        "treatment",
        q2_label,
        "Treatment",
        bins=q2_bins,
        granule_data=granule_data,
        out_dir=out_dir,
        density=density,
        benchling_format=True
    )

if __name__ == "__main__":
    argh.dispatch_command(main)
