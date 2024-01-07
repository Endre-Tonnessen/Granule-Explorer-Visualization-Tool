
# Granule-Explorer-Visualization-Tool

Web application for automating plotting of granule data.  Built in [Shiny for python](https://shiny.posit.co/py/).


To start the application, clone this repository and run:

    shiny run app.py 

When developing add the `--reload` flag for live refreshes/updating.


# Adding plots to the application
Displaying plots has three steps:

1. Creating the plot config
2. Adding a graph_module_ui() function to the 'app_ui' variable
2. Adding a graph_module_server() function to the 'server' function

The plot config states which ui elements should be created and their corresponding ID's. The ´graph_module_ui()´ reads the config and creates ui elements with the specified parameters. The ´graph_module_server()´ reads the ui elements id's and retrieves user inputs from the client to render plots. 

# Plot configuration


### Plot config example:
This example displays all possible elements the config can contain. Each 'input' is associated with a Shiny input function. A full list can be found here. [Shiny UI function reference](https://shiny.posit.co/py/api/ui.input_select.html).

For example one text_input entrie will create one ´ui.input_text()´ element in the UI. Each input entrie can have any parameter the corresponding Shiny function can have. Under the hood, each parameter belonging to an entrie is simply passed on to the created UI element.
```py
overlap_hist_plot_input_options = {
    'plot_type': "histogram", # Type of matplotlib plot this config belongs to. Used to differentiate between plots when downloading internal plot data.
    'allow_internal_data_download': False, # If to allow the download of the internal data in a plot. Warning! This logic has to be implemented for each new plot.
    'allow_multiple_experiments': True, # If plot should accept multiple different experiments. Useful for comparison plots, overlaping histograms etc.
    'text_input': dict({
        #Creates ui.input_text() element with id='plot_label' and value and label as parameters
        "plot_label":dict({
            'value':"Interfacial Tension (N/m)", 
            'label':"X-axis title"
        }),
    }),
    'numeric_input': dict({
        #Creates ui.input_numeric() element with id='n_bins' and value and label as parameters
        "n_bins":dict({
            'value':20, 
            'label':"Nr bins"
        })
    }),
    'bool_input':dict({
        #Creates ui.input_switch() element with id='legend' and value and label as parameters
        'legend':dict({
            'value':False, 
            'label':'legend'
        }),
    }),
    'select_input':dict({
        #Creates ui.input_select() element with id='bin_type' and 'label', 'choices' and 'selected' as parameters
        "bin_type":dict({
            'label':'Bin type', 
            'choices':['count', 'radius', 'log'], 
            'selected':"count"
        }),
    }),
    'select_input_dataset_columns':dict({
        # This is the same as the normal 'select_input' but its 'choices' list will be updated with the column names of the uploaded aggregate data file. 
        # Parameters are ui.input_select() parameters
        "plot_column":dict({
            'label':'X-axis', 
            'choices':['Surface Tension Error (N/m)'], 
            'selected':"Surface Tension Error (N/m)"
        }),
    }),
    'static_input':dict({
        # Any plot parameters that should not create a UI element for the user to interact with, but still has to be specified to the plot function.
        #Plot parameters that do not rely on user input. These will not create ui compenents and are only used server side.
        'plot_title':"Title of plot"
    }),
}
```
