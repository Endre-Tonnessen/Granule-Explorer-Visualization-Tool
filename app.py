import platform
from shiny import App, render, ui, reactive
from shiny.types import ImgData, FileInfo
import shiny.experimental as x
import webbrowser
# from pathlib import Path
from numpy import random 
import pandas as pd
import matplotlib.pyplot as plt
import shinyswatch
import sys
import pathlib
plt2 = platform.system()
if plt2 == 'Windows': pathlib.PosixPath = pathlib.WindowsPath

# Modules
from modules.graph_module import graph_module_ui, graph_module_server
from modules.file_upload_module import file_upload_module_ui, file_upload_module_server

# Plotting tools
import plotting_tools.split_histograms as splth

twoDHist_plot_input_options={
    'plot_type': "histogram",
    'allow_internal_plot_data_download': True,
    'allow_multiple_experiments':False,
    'text_input': dict({
        "plot_title":dict({
            'value':"Interfacial Tension Error (N/m)", 
            'label':"Y-axis title"
        }),
        "row_title":dict({
            'value':"Interfacial Tension (N/m)", 
            'label':"X-axis title"
        }),
    }),
    'numeric_input': dict({}),
    'bool_input':dict({
        #Sliders and such
        'legend':dict({
            'value':False, 
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
        "plot_column":dict({'label':'Y-axis', 'choices':['Interfacial Tension Error (N/m)'], 'selected':"Interfacial Tension Error (N/m)"}),
        "plot_row":dict({'label':'X-axis', 'choices':['Interfacial Tension (N/m)'], 'selected':"Interfacial Tension (N/m)"})
    }),
    'static_input':dict({
        #Inputs that will not change. These will not create ui compenents and are only used server side.
    }),
}
    
scatter_plot_input_options={
    'plot_type': "scatter",
    'allow_internal_plot_data_download': True,
    'allow_multiple_experiments':False,
    'text_input': dict({
        "plot_title":dict({
            'value':"Interfacial Tension Error (N/m)", 
            'label':"Y-axis title"
        }),
        "row_title":dict({
            'value':"Interfacial Tension (N/m)", 
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
        "plot_column":dict({'label':'Y-axis', 'choices':['Interfacial Tension Error (N/m)'], 'selected':"Interfacial Tension Error (N/m)"}),
        "plot_row":dict({'label':'X-axis', 'choices':['Interfacial Tension (N/m)'], 'selected':"Interfacial Tension (N/m)"})
    }),
    'static_input':dict({
        #Inputs that will not change. These will not create ui compenents and are only used server side.
    }),
}

filter_plot_input_options={
    'plot_type': "filter",
    'allow_internal_plot_data_download': True,
    'allow_multiple_experiments':False,
    'text_input': dict({
        "plot_title":dict({
            'value':"Interfacial Tension Error (N/m)", 
            'label':"Y-axis title"
        }),
        "bin_title":dict({
            'value':"Interfacial Tension (N/m)", 
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
        'x_log_scale':dict({
            'value':False, 
            'label':'log_scale'
        }),
        'y_log_scale':dict({
            'value':False, 
            'label':'log_scale'
        }),
        'errors':dict({
            'value':False, 
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
        "plot_column":dict({'label':'Y-axis', 'choices':['Interfacial Tension Error (N/m)'], 'selected':"Interfacial Tension Error (N/m)"}),
        "bin_column":dict({'label':'X-axis', 'choices':['Interfacial Tension (N/m)'], 'selected':"Interfacial Tension (N/m)"})
    }),
    'static_input':dict({
        #Inputs that will not change. These will not create ui compenents and are only used server side.
    }),
}

overlap_hist_plot_input_options={
    'plot_type': "overlap_histogram",
    'allow_internal_plot_data_download': True,
    'allow_multiple_experiments':True, # Allow user to select mutiple experiment
    'text_input': dict({
        # "plot_title":dict({
        #     'value':"Surface Tension Error (N/m)", 
        #     'label':"Y-axis title"
        # }),
        "plot_label":dict({
            'value':"Interfacial Tension (N/m)", 
            'label':"X-axis title"
        }),
    }),
    'numeric_input': dict({
        "n_bins":dict({
            'value':60, 
            'label':"Nr bins"
        }),
        "bin_start":dict({
            'value':-9.5, 
            'label':"Bin start value"
        }),
        "bin_end":dict({
            'value':4, 
            'label':"Bin end value"
        })
    }),
    'bool_input':dict({
        #Sliders and such
        'legend':dict({
            'value':True, 
            'label':'legend'
        }),
        'density':dict({
            'value':False, 
            'label':'density'
        }),
    }),
    'select_input':dict({
        # Custom select inputs. Parameters are ui.input_select() parameters
        "bin_type":dict({'label':'Bin type', 'choices':['linear','log'], 'selected':"log"}),
        # "bin_type":dict({'label':'Bin type', 'choices':['linear', 'geom space', 'log'], 'selected':"geom space"}),
    }),
    'select_input_dataset_columns':dict({
        # Select inputs that are automatically populated with the columns of the dataset
        # Parameters are ui.input_select() parameters
        "plot_column":dict({'label':'X-axis', 'choices':['Interfacial Tension Error (N/m)'], 'selected':"Interfacial Tension Error (N/m)"}),
        # "plot_row":dict({'label':'X-axis', 'choices':['Interfacial Tension (N/m)'], 'selected':"Interfacial Tension (N/m)"})
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
                ui.nav("Overlap Hist", 
                    graph_module_ui(id="overlap_hist", label="Overlap Histogram", plot_input_options=overlap_hist_plot_input_options)
                ),
                ui.nav("Scatter Plot", 
                    graph_module_ui(id="scatteplot", label="Scatter plot", plot_input_options=scatter_plot_input_options)
                ),
                ui.nav("2D Histogram", 
                    graph_module_ui(id="2dhistogram", label="2D Histogram", plot_input_options=twoDHist_plot_input_options)
                ),
                #ui.nav("Filter plot", 
                #    graph_module_ui(id="filter_plot", label="Filter plot", plot_input_options=filter_plot_input_options)
                #),
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
    graph_module_server(id="overlap_hist", granule_data_reactive_value=granule_data_reactive_value, plot_function=splth.overlap_hist, plot_parameters=overlap_hist_plot_input_options) # Pass data to graph module
    graph_module_server(id="scatteplot", granule_data_reactive_value=granule_data_reactive_value, plot_function=splth.scatter_plot, plot_parameters=scatter_plot_input_options) # Pass data to graph module
    graph_module_server(id="2dhistogram", granule_data_reactive_value=granule_data_reactive_value, plot_function=splth.histogram2D, plot_parameters=twoDHist_plot_input_options) # Pass data to graph module
    #graph_module_server(id="filter_plot", 
    #                    granule_data_reactive_value=granule_data_reactive_value, 
    #                    plot_function=splth.filter_plot, 
    #                    plot_parameters=filter_plot_input_options) 

    
        
app = App(ui=app_ui, server=server)
webbrowser.open("http://127.0.0.1:8000", new=2) # Open web browser
