from shiny import App, Inputs, Outputs, Session, module, render, ui, reactive
from shiny.types import ImgData, FileInfo
import pandas as pd
import matplotlib.pyplot as plt
import shiny.experimental as x
import io

from plotting_tools.split_histogram import filter_plot 

@module.ui
def graph_module_ui(label: str, plot_input_options: dict[dict[dict]]):
    # Create text_input elements
    plot_input_text_ui_elements = []
    for k,v in plot_input_options['text_input'].items(): # TODO: Handle KeyError for dictionaries. Prevents UI config from having to include all possible options.
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
            x.ui.layout_column_wrap("450px",
                x.ui.card(
                    x.ui.card_header("Dataset filters"),
                    ui.row( 
                        ui.input_switch(id="sigma_filter_switch", label="Sigma >", width="170px"),
                        ui.input_numeric(id="sigma_filter_input", label="", value=1e-10, step=1e-10, width="200px")
                    ),
                    ui.row(
                        ui.input_switch(id="pass_rate_filter_switch", label="pass_rate >", width="170px"),
                        ui.input_numeric(id="pass_rate_filter_input", label="", value=0.6, step=0.1, width="200px")
                    ),
                    ui.row(
                        ui.input_switch(id="fitting_error_filter_switch", label="fitting_error <", width="170px"),
                        ui.input_numeric(id="fitting_error_filter_input", label="", value=0.5, step=0.1, width="200px")
                    ),
                    ui.row(
                        ui.input_switch(id="fitting_diff_filter_switch", label="fitting_diff >", width="170px"),
                        ui.input_numeric(id="fitting_diff_filter_input", label="", value=0.03, step=0.01, width="200px")
                    ),
                ),
                
            )
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
                elif k == 'select_input_dataset_columns':
                    plot_parameters_from_user_input[k_2] = alias_to_column(input[k_2]()) # Get user input from select elements and transform them back into dataframe column name
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
      
        granule_data_df: list[pd.DataFrame] = granule_data_reactive_value.get()[0] # Call reactive value to get its contents
        
        # Get dataset filters and return filtered data #TODO: Clean up this prototype code block
        query = []
        if input['sigma_filter_switch']():
            sigma_filter = f"sigma > {input['sigma_filter_input']()}"
            query.append(sigma_filter)
        if input['pass_rate_filter_switch']():
            sigma_filter = f"pass_rate > {input['pass_rate_filter_input']()}"
            query.append(sigma_filter)
        if input['fitting_error_filter_switch']():
            sigma_filter = f"fitting_error > {input['fitting_error_filter_input']()}"
            query.append(sigma_filter)
        if input['fitting_diff_filter_switch']():
            sigma_filter = f"fitting_diff > {input['fitting_diff_filter_input']()}"
            query.append(sigma_filter)
        if len(query) > 0:
            query = ''.join(list(map(lambda filter: filter + " and ", query[:-1]))) + query[-1] # Add "and" between queries
            filtered_granule_data: pd.DataFrame = granule_data_df.query(
                query,
                inplace=False
            )
            granule_data_df = filtered_granule_data
        
        return graph_function(granule_data=granule_data_df,**parse_plot_parameters())

    @reactive.Effect
    def update_axies_select():
        """
            Update axis selects with dataframe columns.
            Function is triggered when 'granule_data_reactive_value' is changed (a file is uploaded).
        """
        if not granule_data_reactive_value.is_set(): # Ensure file has been uploaded 
            return        
        
        granule_data_df: list[pd.DataFrame] = granule_data_reactive_value.get() # Call reactive value to get its contents
        column_names: list[str] = granule_data_df[0].columns.to_list()
        filtered_column_names = filter_columns(column_names)                    # Remove blacklisted columns that should not be shown to user.
        column_alias_names: list[str] = columns_to_alias(filtered_column_names) # Get human readable names for df columns
               
        for k,v in plot_parameters['select_input_dataset_columns'].items():
            ui.update_select(id=k,                                      # Update select elements with column aliases
                             choices=column_alias_names, 
                             selected=column_to_alias(v['selected']))   # Get human readable name for the current selected value
    
    @session.download(filename="data.png")
    async def download_plot():  
        """
            File download implemented by yielding bytes, in this case either all at
            once (the entire plot). Filename is determined in the @session.Download decorator ontop of function.
            This determines what the browser will name the downloaded file. 

            TODO: Find alternative approach allowing us to name the download programmatically. Currently it is a static name.
            
            TODO: Place graph creation and formating in its own class.
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


# TODO: Create a new class/file for this logic
column_aliases = { 
                "times":"Times(s)",
                "sigma": "Interfacial Tension (N/m)",
                "kappa_scale": "Bending Rigidity ($k_{\mathrm{B}}T$)",
                "sigma_err": "Surface Tension Error (N/m)",
                "kappa_scale_err": "Bending Rigidity Error($k_{\mathrm{B}}T$)",
                "fitting_error": "Fitting Error",
                "q_2_mag": "$|C_2|^2$",
                "mean_radius":"Mean Radius",
                "pass_rate":"Pass Rate",
                "mean_intensity":"Intensity"}
column_filter = ['granule_id','image_path','x','y','bbox_left','bbox_bottom','bbox_right','bbox_top','figure_path', 'treatment']

def filter_columns(column_names: list[str]) -> list[str]:
    """
        Removes columns user shout not see.
    """
    filtered_column_names = [column for column in column_names if column not in column_filter]
    return filtered_column_names

def columns_to_alias(column_names: list[str]) -> list[str]:
    """
        Returns list of column names replaced with human readable aliases.
        If no alias is found it defaults to returning column name 
    """
    filtered_names = filter_columns(column_names) # Remove columns user should not see

    for i in range(len(filtered_names)):
        if filtered_names[i] in column_aliases.keys():
            filtered_names[i] = column_aliases[filtered_names[i]]
    return filtered_names

def column_to_alias(column_name: str) -> str:
    """
        Returns alias name corresponding to given column. 
        If no alias is found, returns alias.
    """
    if column_name in column_aliases.keys():
        return column_aliases[column_name]
    return column_name

def alias_to_column(alias: str) -> str:
    """ 
        Returns column name corresponding to given alias. 
        If no column is found, returns alias.
    """
    for k,v in column_aliases.items():
        if v == alias:
            return k # Return alias
    return alias # No alias for input
