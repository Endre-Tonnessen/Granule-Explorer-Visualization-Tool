import platform
from shiny import App, render, ui, reactive
from shiny.types import ImgData, FileInfo
import shiny.experimental as x
import webbrowser
from pathlib import Path
from numpy import random 
import pandas as pd
import matplotlib.pyplot as plt
import shinyswatch
import sys
import pathlib
plt = platform.system()
if plt == 'Windows': pathlib.PosixPath = pathlib.WindowsPath
# temp = pathlib.PosixPath
# pathlib.PosixPath = pathlib.WindowsPath

# Modules
from modules.graph_module import graph_module_ui, graph_module_server
from modules.file_upload_module import file_upload_module_ui, file_upload_module_server

# Plotting tools
import plotting_tools.split_histogram as splth

twoDHist_plot_input_options={
    'text_input': dict({
        "plot_title":dict({
            'value':"defualt value", 
            'label':"Y-axis title"
        }),
        "row_title":dict({
            'value':"N/m", 
            'label':"X-axis title"
        }),
    }),
    'numeric_input': dict({}),
    'bool_input':dict({
        #Sliders and such
        'legend':dict({
            'value':True, 
            'label':'legend'
        }),
        'log_scaleX':dict({
            'value':True, 
            'label':'log_scaleX'
        }),
        'log_scaleY':dict({
            'value':True, 
            'label':'log_scaleY'
        }),
    }),
    'select_input':dict({
        # Custom select inputs. Parameters are ui.input_select() parameters
        # "option1":dict({'label':'option1', 'choices':['option1', 'option2'], 'selected':"option1"}),
        # "option2":dict({'label':'option2', 'choices':['option1', 'option2'], 'selected':"option2"}),
    }),
    'select_input_dataset_columns':dict({
        # Select inputs that are automatically populated with the columns of the dataset
        # Parameters are ui.input_select() parameters
        "plot_column":dict({'label':'Y-axis title', 'choices':['Upload file', 'sigma_err'], 'selected':"sigma_err"}),
        "plot_row":dict({'label':'X-axis title', 'choices':['Upload file', 'sigma'], 'selected':"sigma"})
    }),
    'static_input':dict({
        #Inputs that will not change. These will not create ui compenents and are only used server side.
    }),
}
    
scatter_plot_input_options={
    'text_input': dict({
        "plot_title":dict({
            'value':"defualt value", 
            'label':"Y-axis title", 
        }),
        "row_title":dict({
            'value':"N/m", 
            'label':"X-axis title"
        }),
    }),
    'numeric_input': dict({}),
    'bool_input':dict({
        #Sliders and such
        'legend':dict({
            'value':True, 
            'label':'legend'
        }),
        'log_scaleX':dict({
            'value':True, 
            'label':'log_scaleX'
        }),
        'log_scaleY':dict({
            'value':True, 
            'label':'log_scaleY'
        }),
    }),
    'select_input':dict({
        # Custom select inputs. Parameters are ui.input_select() parameters
    }),
    'select_input_dataset_columns':dict({
        # Select inputs that are automatically populated with the columns of the dataset
        # Parameters are ui.input_select() parameters
        "plot_column":dict({'label':'Y-axis title', 'choices':['Upload file', 'sigma_err'], 'selected':"sigma_err"}),
        "plot_row":dict({'label':'X-axis title', 'choices':['Upload file', 'sigma'], 'selected':"sigma"})
    }),
    'static_input':dict({
        #Inputs that will not change. These will not create ui compenents and are only used server side.
    }),
}

filter_plot_plot_input_options={
    'text_input': dict({
        "plot_title":dict({
            'value':"defualt value", 
            'label':"Y-axis title", 
        }),
        "row_title":dict({
            'value':"N/m", 
            'label':"X-axis title"
        }),
    }),
    'numeric_input': dict({
        "n_bins":dict({
            'value':5, 
            'label':"Nr bins"
        })
    }),
    'bool_input':dict({
        #Sliders and such
        'legend':dict({
            'value':True, 
            'label':'legend'
        }),
        'log_scale':dict({
            'value':True, 
            'label':'log_scale'
        }),
        'errors':dict({
            'value':True, 
            'label':'errors'
        }),
    }),
    'select_input':dict({
        # Custom select inputs. Parameters are ui.input_select() parameters
        "bin_type":dict({'label':'Bin type', 'choices':['count', 'radius', 'log'], 'selected':"count"}),
    }),
    'select_input_dataset_columns':dict({
        # Select inputs that are automatically populated with the columns of the dataset
        # Parameters are ui.input_select() parameters
        "plot_column":dict({'label':'Y-axis title', 'choices':['Upload file', 'sigma_err'], 'selected':"sigma_err"}),
        "plot_row":dict({'label':'X-axis title', 'choices':['Upload file', 'sigma'], 'selected':"sigma"})
    }),
    'static_input':dict({
        #Inputs that will not change. These will not create ui compenents and are only used server side.
    }),
}

# UI
app_ui = ui.page_fluid(
        # shinyswatch.theme.spacelab(),
        x.ui.layout_sidebar(
            ui.panel_sidebar(
                file_upload_module_ui("global_file_upload"),
                width=1.3
            ),
        ui.panel_main(
            ui.panel_title("Granule Explorer Visualization Tool", "Granule Explorer"),
            ui.navset_tab(
                # Nav elements
                ui.nav("Scatter Plot", 
                    graph_module_ui(id="scatteplot", label="Scatter plot", plot_input_options=scatter_plot_input_options)
                ),
                ui.nav("2D Histogram", 
                    graph_module_ui(id="2dhistogram", label="2D Histogram", plot_input_options=twoDHist_plot_input_options)
                ),
                ui.nav("Filter plot", 
                    graph_module_ui(id="filter_plot", label="Filter plot", plot_input_options=filter_plot_plot_input_options)
                ),
                ui.nav_menu(
                    "Other links",
                    # body of menu
                    ui.nav("c",

                    ),
                    "Plain text",
                    # create a horizontal line
                    "----",
                    "More text",
                    align="right",
                ),
            ),
        ),
    )
)
   
# Server
def server(input, output, session):
    # Handle file upload
    granule_data_reactive_value: reactive.Value[list[pd.DataFrame]] = file_upload_module_server("global_file_upload")

    # Graph modules
    graph_module_server(id="scatteplot", granule_data_reactive_value=granule_data_reactive_value, graph_function=splth.scatter_plot, plot_parameters=scatter_plot_input_options) # Pass data to graph module
    graph_module_server(id="2dhistogram", granule_data_reactive_value=granule_data_reactive_value, graph_function=splth.histogram2D, plot_parameters=twoDHist_plot_input_options) # Pass data to graph module
    graph_module_server(id="filter_plot", granule_data_reactive_value=granule_data_reactive_value, graph_function=splth.filter_plot, plot_parameters=filter_plot_plot_input_options) # Pass data to graph module

    
        
app = App(ui=app_ui, server=server)
# webbrowser.open("http://127.0.0.1:8000", new=2)
