import pandas as pd
from shiny import App, Inputs, Outputs, Session, module, render, ui, reactive
import io
import matplotlib.pyplot as plt
from typing import Callable, Tuple, Union

def create_fig(input: Inputs, 
               granule_data_df: pd.DataFrame, 
               plot_function: Callable, 
               plot_parameters: dict) -> Tuple[plt.figure, pd.DataFrame]:
    """Based on given data, graph function and graph parameters, returns resulting figure from the filtered dataset along with the plot_data.

    Parameters
    ----------
    input: Inputs
        Shiny variable containing user input

    granule_data_df: pd.DataFrame
        Data used in plot creation

    plot_function: Callable
        Function creating the plot

    plot_parameters: dict
        user parameters passed on to the plot function 

    Returns:
    ----------
        Returns 2-d tuple of (fig: matplotlib.figure, plot_data: pd.Dataframe)
    """
    # If multiple experiments are selected, selectize will return a list of strings. If only 1 experiment, then just one str
    # Corresponds to selectizes parameter "multiple" being True or False.
    selected_experiments: tuple[str] | str = input['experiment_selectize_input']()

    # Filter data based on user selected filter
    granule_data_df = filter_dataset(input, granule_data_df)
 
    # If selected_experiments is not a tuple, add "plit_group" parameter. 
    # Telling plot funtion to only group by the given experiment. 
    if type(selected_experiments) is not tuple: 
        fig, plot_data_df = plot_function(granule_data=granule_data_df, 
                            group_by="experiment", 
                            plot_group=selected_experiments, 
                            save_png=False, 
                            **plot_parameters)
    else: # If multiple experiments, omit plot_group parameter. Used for the overlap_hist plot.
        granule_data_df = granule_data_df[granule_data_df["experiment"].isin(selected_experiments)]
        fig, plot_data_df = plot_function(granule_data=granule_data_df, 
                            group_by="experiment", 
                            save_png=False, 
                            **plot_parameters)
    
    return fig, plot_data_df

def create_download_figure(input: Inputs, 
                           granule_data_df: pd.DataFrame, 
                           plot_function: Callable, 
                           plot_parameters: dict, 
                           save_buffer: io.BytesIO,
                           filetype: str) -> plt.figure:
    """
    Creates plot with ouput settings. Saves to given io buffer zone for download in browser.
    Returns the figure.

    Parameters
    ----------
    input: Inputs 
        Shiny variable containing user input
    
    granule_data_df: pd.DataFrame
        Data used in plot creation
    
    plot_function: Callable
        Function creating the plot
    
    plot_parameters: dict
        user parameters passed on to the plot function 
    
    save_buffer: io.BytesIO
        buffer the figure is save to for IO operations
    
    filetype: str
        Filetype of output plot. Either "svg" or "png"

    Returns
    -------
        figure: matplotlib.figure
        
        Returns the fig due to the downloading internal plot data feature.
        Its side-effect is saving the created figure in the bytes buffer.
    """
    fig_and_data: Tuple[plt.figure, pd.DateOffset] = create_fig(input=input, 
                     granule_data_df=granule_data_df, 
                     plot_function=plot_function, 
                     plot_parameters=plot_parameters)
    fig: plt.figure = fig_and_data[0] # Grab just the figure
    
    # Get user input from UI
    padding = input['download_figure_padding']()
    tl_padding = input['download_figure_tl_padding']()
    despine = input['download_figure_despine_axis']()
    dpi = input['download_figure_dpi']()
    plot_height = input['download_figure_height_inches']()
    plot_width = input['download_figure_width_inches']()
    

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
    fig.set_size_inches(plot_width, plot_height)
    fig.savefig(save_buffer, dpi=dpi, format=filetype, **plotKwargs)
    return fig


def filter_dataset(input: Inputs, granule_data_df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters given dataset based on user selected values.
    Returns a new Dataframe.
    
    Parameters
    ----------
    input: Inputs
        Shiny variable containing user input

    granule_data_df: pd.DataFrame
        Dataframe to filter

    Returns
    -------
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
        # Add 'and' between all queries except the last one.
        query = ''.join(list(map(lambda filter: filter + " and ", query[:-1]))) + query[-1]
        filtered_granule_data: pd.DataFrame = granule_data_df.query(
            query,
            inplace=False
        )
        granule_data_df = filtered_granule_data
    return granule_data_df
