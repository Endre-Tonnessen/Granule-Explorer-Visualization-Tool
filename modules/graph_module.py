from shiny import App, Inputs, Outputs, Session, module, render, ui, reactive
from shiny.types import ImgData, FileInfo
import pandas as pd
import matplotlib.pyplot as plt
import shiny.experimental as x
import io
from typing import Callable

from plotting_tools.split_histogram import filter_plot 
from plotting_tools.create_plot import create_download_figure, create_fig

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
                                    # ui.download_button("download_plot_png", "Download Plot"),
                                    ui.input_action_button("modal_download", "Download")),
                ui.hr(),

                # Unpack ui elemets from list
                ui.input_selectize(id="treatment_selectize_input", label="Select treatments", choices=[""], multiple=True, width="200px"),
                *plot_input_select_axis_ui_elements,
                *plot_input_switch_ui_elements,
                
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
            ui.row(
                # Filter
                x.ui.layout_column_wrap("450px",
                    x.ui.card(
                        x.ui.card_header("Dataset filters"),
                        ui.row( 
                            ui.input_switch(id="sigma_filter_switch", label="Surface Tension >", width="200px", value=True), # Sigma
                            ui.input_numeric(id="sigma_filter_input", label="", value=1e-10, step=1e-10, width="200px")
                        ),
                        ui.row(
                            ui.input_switch(id="pass_rate_filter_switch", label="Pass Rate >", width="200px", value=True),
                            ui.input_numeric(id="pass_rate_filter_input", label="", value=0.6, step=0.1, width="200px")
                        ),
                        ui.row(
                            ui.input_switch(id="fitting_error_filter_switch", label="Fitting Error <", width="200px", value=True),
                            ui.input_numeric(id="fitting_error_filter_input", label="", value=0.5, step=0.1, width="200px")
                        ),
                        ui.row(
                            ui.input_switch(id="fitting_diff_filter_switch", label="Fitting diff >", width="200px", value=True),
                            ui.input_numeric(id="fitting_diff_filter_input", label="", value=0.03, step=0.01, width="200px")
                        ),
                        # max_height="400px",
                        fill=False
                    ),

                    x.ui.card(
                        x.ui.card_header("Plot parameters"),
                        ui.page_bootstrap(
                            *plot_input_text_ui_elements, # Unpack plot input ui elements
                            *plot_input_select_ui_elements,
                            *plot_input_numeric_ui_elements,
                        ),
                    )
                )
             
            
            )
        )
    )


@module.server
def graph_module_server(input: Inputs,
                        output: Outputs,
                        session: Session, 
                        granule_data_reactive_value: reactive.Value[pd.DataFrame], 
                        plot_function: Callable, 
                        plot_parameters: dict[dict[dict]]):
    
    def parse_plot_parameters() -> dict:
        """Parses and returns a 1d dictonary with the plot parameters required for the plot function.
                -> Any non-static elements value is retrieved from the ui. 

        Returns:
            dict: Dictionary with key value pairs for the plotting function
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
    @render.plot(alt="Plot")
    @reactive.event(input.update_plot, input.treatment_selectize_input) 
    def plot():
        """
            Renders a new plot based on the given plot function and its plot-parameters.
            If a file is uploaded or the "Update plot" button is triggered, this function will run.
        """
        if not granule_data_reactive_value.is_set(): # Ensure file has been uploaded 
            return
      
        granule_data_df: pd.DataFrame = granule_data_reactive_value.get() # Call reactive value to get its contents
        return create_fig(input=input, 
                          granule_data_df=granule_data_df, 
                          plot_function=plot_function,
                          plot_parameters=parse_plot_parameters())
        
    @reactive.Effect
    def update_axies_select(): 
        """
            Update axis selects with dataframe columns.
            Function is triggered when 'granule_data_reactive_value' is changed (a file is uploaded).
        """
        if not granule_data_reactive_value.is_set(): # Ensure file has been uploaded 
            return        
        
        granule_data_df: pd.DataFrame = granule_data_reactive_value.get() # Call reactive value to get its contents
        column_names: list[str] = granule_data_df.columns.to_list()
        filtered_column_names: list[str] = filter_columns(column_names)         # Remove blacklisted columns that should not be shown to user.
        column_alias_names: list[str] = columns_to_alias(filtered_column_names) # Get human readable names for df columns
               
        for k,v in plot_parameters['select_input_dataset_columns'].items():
            ui.update_select(id=k,                                      # Update select elements with column aliases
                             choices=column_alias_names, 
                             selected=column_to_alias(v['selected']))   # Get human readable name for the current selected value
    
    @reactive.Effect
    def update_axis_name_text_input():
        """Updates x and y-axis names based on selected columns #TODO: Add error handling for plots not using 'select_input_dataset_columns'. Will it fail if element not found?
        """
        if not granule_data_reactive_value.is_set(): # Ensure file has been uploaded 
            return 
        ui.update_text(id='plot_title', value=input.plot_column())
        ui.update_text(id='row_title', value=input.plot_row())

    @reactive.Effect 
    # @reactive.event(granule_data_reactive_value)
    def update_treatment_selectize_input():
        if not granule_data_reactive_value.is_set(): # Ensure file has been uploaded 
            return 
        granule_data_df: pd.DataFrame = granule_data_reactive_value.get() # Call reactive value to get its contents
        
        choices: list[str] = granule_data_df['treatment'].unique().tolist()
        # print(choices)
        ui.update_selectize(id="treatment_selectize_input", choices=choices, selected=choices)

    @reactive.Effect
    @reactive.event(input.modal_download)
    def modal_download():
        m = ui.modal(
            ui.row(
                ui.column(6, 
                      ui.input_numeric(id="download_figure_dpi", label="Dpi", value=300, width="100px"),
                      ui.input_numeric(id="download_figure_padding", label="Padding", value=0.15, width="100px"),
                      ui.input_numeric(id="download_figure_tl_padding", label="Tl padding", value=1.08, width="100px"),
                      ),
                ui.column(6, 
                    #   ui.input_select(id="download_file_format", choices=["png", "svg", "jpeg"], selected="png", label="", width="100px"),
                      ui.input_switch(id="download_figure_despine_axis", label="Despine axis"),
                      ui.input_numeric(id="download_figure_height_inches", label="Height (inches)", value=5, width="100px"),
                      ui.input_numeric(id="download_figure_width_inches", label="Weight (inches)", value=8, width="100px"),
                ),
            ),
            ui.download_button("download_plot_png", "Download png"),
            ui.download_button("download_plot_svg", "Download svg"),
            title="Download config",
            easy_close=True,
            footer=None,
        )
        ui.modal_show(m)

    @session.download(filename="plot.png")
    async def download_plot_png():  
        """
            File download implemented by yielding bytes, in this case either all at
            once (the entire plot). Filename is determined in the @session.Download decorator ontop of function.
            This determines what the browser will name the downloaded file.     
        """
        if not granule_data_reactive_value.is_set(): # Ensure file has been uploaded 
                return
        
        with io.BytesIO() as buf:
            granule_data_df: pd.DataFrame = granule_data_reactive_value.get()
            fig = create_download_figure(input=input, 
                                         granule_data_df=granule_data_df, 
                                         plot_function=plot_function, 
                                         plot_parameters=parse_plot_parameters(),
                                         save_buffer=buf,
                                         filetype="png")
            yield buf.getvalue()
            plt.close(fig=fig)

    @session.download(filename="plot.svg")
    async def download_plot_svg():  
        if not granule_data_reactive_value.is_set(): # Ensure file has been uploaded 
                return
        
        with io.BytesIO() as buf:
            granule_data_df: pd.DataFrame = granule_data_reactive_value.get()
            fig = create_download_figure(input=input, 
                                         granule_data_df=granule_data_df, 
                                         plot_function=plot_function, 
                                         plot_parameters=parse_plot_parameters(),
                                         save_buffer=buf,
                                         filetype="svg")
            yield buf.getvalue()
            plt.close(fig=fig)





# TODO: Create a new class/file for this logic
column_aliases = { 
                "times":"Times(s)",
                "sigma": "Interfacial Tension (N/m)",
                "kappa_scale": "Bending Rigidity",
                "sigma_err": "Surface Tension Error (N/m)",
                "kappa_scale_err": "Bending Rigidity Error",
                "fitting_error": "Fitting Error",
                "q_2_mag": "$|C_2|^2$",
                "mean_radius":"Mean Radius",
                "pass_rate":"Pass Rate",
                "mean_intensity":"Intensity"}
column_filter = ['granule_id','image_path','x','y','bbox_left','bbox_bottom','bbox_right','bbox_top','figure_path', 'treatment']

def filter_columns(column_names: list[str]) -> list[str]:
    """
        Removes columns user should not see.
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
        If no alias is found, returns column_name.
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
