import pandas as pd
from pathlib import Path

from shiny import App, Inputs, Outputs, Session, render, ui, module, reactive
import shiny.experimental as x
from shiny.types import FileInfo
# from ..plotting_tools.split_histogram import read_data

"""
    Module for handling file uploads, returning the .h5 file to the global context so it may be used by the graphing modules to create plots. 
"""
@module.ui
def file_upload_module_ui():
    return (
        x.ui.tooltip(
            ui.input_file("graunle_aggregate_data", "Upload granule data", accept=[".h5"], multiple=True),
            "Upload aggregate_data.h5 files to begin!",
            id="graunle_aggregate_data_upload_tool_tip",
            options={
                "show":True
            },
            show=True
        ),
        ui.input_file("graunle_image_data", "Upload image data", accept=[".ims"], multiple=False)
    )
        
@module.server
def file_upload_module_server(input: Inputs, output: Outputs, session: Session) -> reactive.Value[pd.DataFrame]:
    uploaded_file = reactive.Value()

    @reactive.Effect
    @reactive.event(input.graunle_aggregate_data)
    def set_uploaded_file():
        """
            Reads and formats the uploaded .h5 files. 
            Sets the results to the reactive value container.
        """
        f: list[FileInfo] = input.graunle_aggregate_data()
        # Get path to each uploaded file
        file_paths: list[Path] = [Path(f[i]["datapath"]) for i in range(len(f))]
        df = read_data(file_paths)
        uploaded_file.set(df)

    
    return uploaded_file


#TODO: Import this from plot_tools?

import re
# Regex capturing --N group termiated with '--' or '.'
re_experiment_name = re.compile("--N(?P<exp>[\d\w_-]+?)(?:--|\.)")
re_time_stamp = re.compile("_(.+)--N")


def read_data(input_file: list[Path], data_file_name="aggregate_fittings.h5"):
    """
    Read and format the data in the given paths. If multiple files are uploaded it will concatenate them.
    Args:
        input_file (list[Path]): _description_
        data_file_name (str, optional): _description_. Defaults to "aggregate_fittings.h5".

    Returns:
        _type_: _description_
    """

    if len(input_file) > 1:       
        granule_data = pd.concat(map(_load_terms, input_file), ignore_index=True)
    else:
        granule_data = _load_terms(input_file[0])

    granule_data["sigma_diff"] = 10000000.0
    granule_data["fitting_diff"] = 10000000.0

    return granule_data


def _load_terms(aggregate_fittings_path: Path) -> pd.DataFrame:
    """Load the spectrum fitting terms and physical values from disk."""
    # Container for the physical properties of the granules
    # print(aggregate_fittings_path)
    aggregate_fittings = pd.read_hdf(
        aggregate_fittings_path, key="aggregate_data", mode="r"
    )

    aggregate_fittings["treatment"] = _get_treament_type(
        aggregate_fittings["figure_path"].iloc[0]
    )

    try:
        times = [_convert_to_sec(path) for path in aggregate_fittings["figure_path"]]

        start = min(times)
        times = [time - start for time in times]
    except:
        times = 1.0

    aggregate_fittings["times"] = times

    return aggregate_fittings


def _get_treament_type(im_path):
    """Get the treatment name from the image path."""
    path_name = Path(im_path).name
    experiment_group = re_experiment_name.search(path_name)

    if experiment_group is None:
        # print(path_name)
        # raise ValueError("No regex match")
        return "unknown"

    experiment_name = experiment_group.groupdict()["exp"]
    # print(path_name, " ", experiment_name)
    if experiment_name.startswith("Control") or experiment_name.startswith("As"):
        return "As"
    if experiment_name.startswith("Cz"):
        return "Cz"
    if experiment_name.startswith("FXR1"):
        if experiment_name.endswith("mCh"):
            return "FXR1-G3BP1"
        elif experiment_name.endswith("GFP"):
            return "FXR1-FXR1"
        else:
            return "NaAs+FXR1"
    if experiment_name.startswith("Caprin"):
        return "NaAs+Caprin1"
    raise ValueError("Unable to get experiment name.")

def _convert_to_sec(path):
    time = re_time_stamp.findall(path)
    t = time[0].split(".")
    return int(t[0]) * 3600 + int(t[1]) * 60 + int(t[2])