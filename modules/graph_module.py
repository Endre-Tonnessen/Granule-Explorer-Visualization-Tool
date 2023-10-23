from shiny import App, Inputs, Outputs, Session, module, render, ui, reactive
from shiny.types import ImgData, FileInfo
import pandas as pd
import matplotlib.pyplot as plt
import shiny.experimental as x
import io
import os
import sys
from pathlib import Path

from plotting_tools.split_histogram import filter_plot 

@module.ui
def graph_module_ui(label: str, plot_input_options):
    # Create text_input elements
    plot_input_text_ui_elements = []
    for k,v in plot_input_options['text_input'].items():
        plot_input_text_ui_elements.append(ui.input_text(id=k, **v))

    # Create select elements that will be updated by the update_axies_select(). Adding dataset column names to the 'choices' parameter.
    plot_input_select_axis_ui_elements = []
    for k, v in plot_input_options['select_input_dataset_columns'].items():
        plot_input_select_axis_ui_elements.append(ui.input_select(id=k, **v))

    # Create normal select elements without any serverside update function 
    plot_input_select_ui_elements = []
    for k, v in plot_input_options['select_input'].items():
        plot_input_select_ui_elements.append(ui.input_select(id=k, **v))

    # Create switch elements
    plot_input_switch_ui_elements = []
    for k, v in plot_input_options['bool_input'].items():
        plot_input_switch_ui_elements.append(ui.input_switch(id=k, **v))

    # Create switch elements
    plot_input_numeric_ui_elements = []
    for k, v in plot_input_options['numeric_input'].items():
        plot_input_numeric_ui_elements.append(ui.input_numeric(id=k, **v))

    return ui.row(
        ui.row(
            ui.column(4,
                ui.row(),
                
                ui.page_bootstrap(ui.input_action_button("update_plot", "Update plot"),
                                    ui.download_button("download_plot", "Download Plot")),
                ui.hr(),

                # Unpack ui elemets from list
                *plot_input_select_axis_ui_elements,
                *plot_input_switch_ui_elements,
                *plot_input_numeric_ui_elements
            ),
            ui.column(8,
                x.ui.card(
                    x.ui.card_header(label),
                    ui.output_plot("plot", click=True)
                ),
                # {"style": "background-color: #eee;"}
            )
    ),
        ui.row(
             ui.page_bootstrap(
                *plot_input_text_ui_elements, # Unpack plot input ui elements
                *plot_input_select_ui_elements
            ),
            ui.output_data_frame("contents"),
        )
    )


@module.server
def graph_module_server(input: Inputs,
                        output: Outputs,
                        session: Session, 
                        granule_data_reactive_value: reactive.Value[list[pd.DataFrame]], 
                        graph_function, 
                        plot_parameters: dict[dict[dict]]):
    
    def parse_plot_parameters() -> dict:
        """
            Parses and returns a 1d dictonary with the plot parameters required for the graph function.
                -> Any non-static element is retrieved from the ui.  
        """
        plot_parameters_from_user_input = dict()
        # Update user input values from corresponding ui input elements. k_2 is the id for each input in ui.
        for k, _ in plot_parameters.items():
            for k_2, v_2 in plot_parameters[k].items():
                if k == "static_input": # If static value, no need to get it from ui
                    plot_parameters_from_user_input[k_2] = v_2['value']
                else:
                    plot_parameters_from_user_input[k_2] = input[k_2]() # Get user input from ui
        return plot_parameters_from_user_input
    
    @output
    @render.plot(alt="A simulation plot")
    @reactive.event(input.update_plot, granule_data_reactive_value)
    def plot():
        """
            Renders a new plot based on the given graph function and its plot-parameters.
            If a file is uploaded or the "Update plot" button is triggered, this function will run.
        """
        if not granule_data_reactive_value.is_set(): # Ensure file has been uploaded 
            return
      
        granule_data_df: list[pd.DataFrame] = granule_data_reactive_value.get() # Call reactive value to get its contents
        return graph_function(granule_data=granule_data_df[0],**parse_plot_parameters())

    @reactive.Effect
    def update_axies_select():
        """
            Update axis selects with dataframe columns.
            Function is triggered when 'granule_data_reactive_value' is changed (a file is uploaded).
        """
        if not granule_data_reactive_value.is_set(): # Ensure file has been uploaded 
            return        
        granule_data_df: list[pd.DataFrame] = granule_data_reactive_value.get() # Call reactive value to get its contents

        for k,v in plot_parameters['select_input_dataset_columns'].items():
            ui.update_select(id=k, choices=granule_data_df[0].columns.to_list(), selected=v['selected'])
    
    @session.download(filename="data.png")
    async def download_plot():  
        """
            File download implemented by yielding bytes, in this case either all at
            once (the entire plot). Filename is determined in the @session.Download decorator ontop of function.
            This determines what the browser will name the downloaded file. 

            TODO: Find alternative approach allowing us to name the download programmatically. Currently it is a static name.
        """
        if not granule_data_reactive_value.is_set(): # Ensure file has been uploaded 
                return
        with io.BytesIO() as buf:
            granule_data_df: list[pd.DataFrame] = granule_data_reactive_value.get()
            fig = graph_function(granule_data=granule_data_df[0], **parse_plot_parameters())
            padding=0.15
            tl_padding=1.08
            despine=True
            dpi=330

            # Remove the right and top axis
            axs = fig.get_axes()
            if despine:
                [despine_axis(ax) for ax in axs]

            plotKwargs = {}
            if padding:
                plotKwargs = dict(bbox_inches="tight", pad_inches=padding)

            fig.tight_layout(pad=tl_padding)
            fig.savefig(buf, dpi=dpi, format="png", **plotKwargs)
            yield buf.getvalue()
            plt.close(fig=fig)

def despine_axis(ax):
    """Remove the top and right axis.

    This emulates seaborn.despine, but doesn't require the modules.
    """
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)