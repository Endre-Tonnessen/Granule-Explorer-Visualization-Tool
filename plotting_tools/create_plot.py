import pandas as pd
from shiny import App, Inputs, Outputs, Session, module, render, ui, reactive
import io



"""
    Class handles all plot creation logic, returning the figure

"""



def create_fig(input: Inputs, 
               granule_data_df: pd.DataFrame, 
               graph_function, 
               plot_parameters: dict):
    """
    
    """
    granule_data_df = filter_dataset(input, granule_data_df)
    fig = graph_function(granule_data=granule_data_df, **plot_parameters)
    return fig

def create_download_figure(input: Inputs, 
                           granule_data_df: pd.DataFrame, 
                           graph_function, 
                           plot_parameters: dict, 
                           save_buffer: io.BytesIO):
    """
        Creates plot with ouput settings.. 
        Saves to given io buffer zone for download in browser.
    """
    fig = create_fig(input=input, 
                     granule_data_df=granule_data_df, 
                     graph_function=graph_function, 
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


def filter_dataset(input: Inputs, granule_data_df: pd.DataFrame):
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
