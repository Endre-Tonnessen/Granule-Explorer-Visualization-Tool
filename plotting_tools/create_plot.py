import pandas as pd
from shiny import App, Inputs, Outputs, Session, module, render, ui, reactive
import io
import matplotlib.pyplot as plt
from typing import Callable

def create_fig(input: Inputs, 
               granule_data_df: pd.DataFrame, 
               plot_function: Callable, 
               plot_parameters: dict) -> plt.figure:
    """Based on given data, graph function and graph parameters, returns resulting figure from the filtered dataset.

    Args:
        input (Inputs): Shiny variable containing user input
        granule_data_df (pd.DataFrame): Data used in plot creation
        plot_function (Callable): Function creating the plot
        plot_parameters (dict): user parameters passed on to the plot function 

    Returns:
        matplotlib.figure.Figure: Figure
    """
    granule_data_df = filter_dataset(input, granule_data_df)
    fig = plot_function(granule_data=granule_data_df, **plot_parameters)
    return fig

def create_download_figure(input: Inputs, 
                           granule_data_df: pd.DataFrame, 
                           plot_function: Callable, 
                           plot_parameters: dict, 
                           save_buffer: io.BytesIO):
    """Creates plot with ouput settings. Saves to given io buffer zone for download in browser.

    Args:
        input (Inputs): Shiny variable containing user input
        granule_data_df (pd.DataFrame): Data used in plot creation
        plot_function (Callable): Function creating the plot
        plot_parameters (dict): user parameters passed on to the plot function 
        save_buffer (io.BytesIO): buffer the figure is save to for IO operations

    Returns:
        This function does not return anything.  
        Its side-effect is saving the created figure in the bytes buffer.
    """
    fig = create_fig(input=input, 
                     granule_data_df=granule_data_df, 
                     plot_function=plot_function, 
                     plot_parameters=plot_parameters)
    # Output settings
    padding=0.15
    tl_padding=1.08
    despine=True
    dpi=330

    def despine_axis(ax):
        """Remove the top and right axis.

        This emulates seaborn.despine, but doesn't require the modules.
        """
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

    # Remove the right and top axis
    axs = fig.get_axes()
    if despine:
        [despine_axis(ax) for ax in axs]

    plotKwargs = {}
    if padding:
        plotKwargs = dict(bbox_inches="tight", pad_inches=padding)

    fig.tight_layout(pad=tl_padding)
    fig.savefig(save_buffer, dpi=dpi, format="png", **plotKwargs)


def filter_dataset(input: Inputs, granule_data_df: pd.DataFrame) -> pd.DataFrame:
    """Filters given dataset based on user selected values.
       Returns a new Dataframe.
    Args:
        input (Inputs): Shiny variable containing user input
        granule_data_df (pd.DataFrame): Dataframe to filter

    Returns:
        pd.DataFrame: Filtered dataset
    """
    # Get dataset filters and return filtered data #TODO: Clean up this prototype code block
    query = []
    if input['sigma_filter_switch']():
        sigma_filter = f"sigma > {input['sigma_filter_input']()}"
        query.append(sigma_filter)
    if input['pass_rate_filter_switch']():
        pass_rate_filter = f"pass_rate > {input['pass_rate_filter_input']()}"
        query.append(pass_rate_filter)
    if input['fitting_error_filter_switch']():
        fitting_error_filter = f"fitting_error > {input['fitting_error_filter_input']()}"
        query.append(fitting_error_filter)
    if input['fitting_diff_filter_switch']():
        fitting_diff_filter = f"fitting_diff > {input['fitting_diff_filter_input']()}"
        query.append(fitting_diff_filter)

    # If any filters, run query
    if len(query) > 0:
        query = ''.join(list(map(lambda filter: filter + " and ", query[:-1]))) + query[-1] # Add "and" between queries
        filtered_granule_data: pd.DataFrame = granule_data_df.query(
            query,
            inplace=False
        )
        granule_data_df = filtered_granule_data
    return granule_data_df
