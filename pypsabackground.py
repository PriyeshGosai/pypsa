# Toolbox for running and interacting with PyPSA using Google Colab.
# Developed by Priyesh Gosai - priyesh.gosai@gmail.com
# Version 2 26 February 2024

import os
import pandas as pd
from ipywidgets import interact, fixed
from IPython.display import display, clear_output
import ipywidgets as widgets
from datetime import datetime, timedelta
import shutil
import logging

# Check if the script is running in Google Colab
if 'COLAB_GPU' in os.environ:
    # Environment-specific installation commands
    os.system('pip install pypsa')
    os.system('pip install -q pyomo')
    os.system('apt-get install -y -qq glpk-utils')
    os.system('pip install highspy')
    os.system('pip install --upgrade gdown')
    from google.colab import drive
    from google.colab import output
else:
    print("Not running in Google Colab. Skipping installations.")

# Functions for running PyPSA

def convert_excel_to_csv(excel_file_path, csv_folder_path):
    """
    Converts each sheet in an Excel file to a CSV file, only for sheets whose names are in a predefined list. 
    The function checks if the target folder exists, and only specific CSV files related to the Excel file's 
    sheets are deleted and recreated.

    Parameters:
    excel_file_path (str): The file path of the Excel file.
    csv_folder_path (str): The path to the folder where CSV files will be saved.

    Returns:
    List[str]: Paths to the successfully created CSV files.
    """
    logging.basicConfig(level=logging.INFO)
    components = ['stores', 'generators', 'buses', 'carriers', 'generators-p_set', 'links', 'loads', 'loads-p_set', 'snapshots','network','stores-e_min_pu']
    created_csv_files = []

    # Ensure the CSV folder exists
    if not os.path.exists(csv_folder_path):
        os.makedirs(csv_folder_path)
    else:
        # Clear only relevant CSV files in the folder
        for item in os.listdir(csv_folder_path):
            if item.endswith(".csv") and item.replace(".csv", "") in components:
                os.remove(os.path.join(csv_folder_path, item))

    try:
        xls = pd.ExcelFile(excel_file_path)
        for sheet_name in xls.sheet_names:
            if sheet_name in components:
                df = xls.parse(sheet_name)
                csv_file_path = os.path.join(csv_folder_path, f"{sheet_name}.csv")
                df.to_csv(csv_file_path, index=False)
                created_csv_files.append(csv_file_path)
                logging.info(f"Converted {sheet_name} to CSV.")
    except Exception as e:
        logging.error(f"Error converting Excel to CSV: {e}")
        return []

    logging.info(f"Conversion complete. CSV files are saved in '{csv_folder_path}'")
    return csv_folder_path #created_csv_files

def postprocess_network_results(network):
    # Postprocessing for Generators
    GeneratorList = [idx for idx in network.generators.index if idx.startswith("Gen-")] + \
                    [idx for idx in network.links.index if idx.startswith("Gen-")]
    GeneratorNames = [idx.replace("Gen-", "") for idx in GeneratorList]
    df_gen_existing = network.generators_t.p
    df_link_existing = network.links_t.p2*-1
    df_Generator = pd.DataFrame(0.0, index=df_gen_existing.index, columns=GeneratorList)  # Note the 0.0 for float
    df_Generator.update(df_gen_existing)
    df_Generator.update(df_link_existing)
    df_Generator.columns = GeneratorNames
    # Postprocessing Spill lines
    SpillList = [idx for idx in network.links.index if idx.startswith("Spl-")]
    SpillNames = [idx.replace("Spl-", "") for idx in SpillList]
    df_spill_existing = network.links_t.p0
    df_Spill = pd.DataFrame(0.0, index=df_spill_existing.index, columns=SpillList)
    df_Spill.update(df_spill_existing)
    df_Spill.columns = SpillNames
    # Postprocessing Transmission lines
    TransmissionLineList = [idx for idx in network.links.index if idx.startswith("Tx-")]
    TransmissionLineNames = [idx.replace("Tx-", "") for idx in TransmissionLineList]
    df_tx_existing = network.links_t.p0
    df_transmission = pd.DataFrame(0.0, index=df_tx_existing.index, columns=TransmissionLineList)
    df_transmission.update(df_tx_existing)
    df_transmission.columns = TransmissionLineNames
    # Postprocessing River inflow
    RiverInflows = [idx for idx in network.generators.index if idx.startswith("RvrIn")]
    RiverInflowNames = [idx.replace("RvrIn ", "") for idx in RiverInflows]
    df_RiverIn_existing = network.generators_t.p
    df_RiverIn = pd.DataFrame(0.0, index=df_RiverIn_existing.index, columns=RiverInflows)
    df_RiverIn.update(df_RiverIn_existing)
    df_RiverIn.columns = RiverInflowNames
    # Postprocessing River outflow
    RiverOutflows = [idx for idx in network.generators.index if idx.startswith("RvrOut")]
    RiverOutflowNames = [idx.replace("RvrOut ", "") for idx in RiverOutflows]
    df_RiverOut_existing = network.generators_t.p
    df_RiverOut = pd.DataFrame(0.0, index=df_RiverOut_existing.index, columns=RiverOutflows)
    df_RiverOut.update(df_RiverOut_existing)
    df_RiverOut.columns = RiverOutflowNames
    # New dataframe with the loads
    total_demand = network.loads_t.p
    # Dam levels and flows
    dam_level = network.stores_t.e
    dam_flows = network.stores_t.p

    # Return all processed dataframes
    return {
        'Generator Data': df_Generator,
        'Spill Data': df_Spill,
        'Transmission Data': df_transmission,
        'River Inflow Data': df_RiverIn,
        'River Outflow Data': df_RiverOut,
        'Total Demand': total_demand,
        'Dam Level': dam_level,
        'Dam Flows': dam_flows
    }

from datetime import datetime, timedelta

def calculate_end_date(start_date_str, days=7):
    """
    Calculate the end date given a start date and the number of days to add.
    
    The input start date can be in 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM' format.
    If only the date is provided, the time is assumed to be 00:00.
    The output will be in 'YYYY-MM-DD HH:MM' format, including hours and minutes.
    
    Parameters:
    - start_date_str (str): The start date in 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM' format.
    - days (int): The number of days to add to the start date.
    
    Returns:
    - str: The end date in 'YYYY-MM-DD HH:MM' format.
    """
    # Try to parse the input string assuming it includes hours and minutes
    try:
        start_date = datetime.strptime(start_date_str, '%Y-%m-%d %H:%M')
    except ValueError:
        # If parsing fails, assume the string is in 'YYYY-MM-DD' format and set time to 00:00
        start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
        start_date = start_date.replace(hour=0, minute=0)  # Set time to 00:00

    # Calculate the end date by adding the specified number of days
    end_date = start_date + timedelta(days=days)

    # Convert the end date back to a string in 'YYYY-MM-DD HH:MM' format
    return end_date.strftime('%Y-%m-%d %H:%M')

# Example usage:
# print(calculate_end_date("2024-03-25", 10))  # Input without time
# print(calculate_end_date("2024-03-25 14:30", 10))  # Input with time



# from datetime import datetime, timedelta

# def calculate_end_date(start_date_str, days=7):
#     """
#     Calculate the end date given a start date and the number of days to add.
    
#     This version of the function aims to be region-independent by treating dates as timezone-naive.
    
#     Parameters:
#     - start_date_str (str): The start date in 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS' format.
#     - days (int): The number of days to add to the start date.
    
#     Returns:
#     - str: The end date in 'YYYY-MM-DD' format.
#     """
#     # Attempt to parse the start date string with time information
#     try:
#         start_date = datetime.strptime(start_date_str, '%Y-%m-%d %H:%M:%S')
#     except ValueError:
#         # If it fails, assume the string is in 'YYYY-MM-DD' format
#         start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
    
#     # Calculate the end date by adding the specified number of days
#     end_date = start_date + timedelta(days=days)
    
#     # Convert the end date back to a string in 'YYYY-MM-DD' format
#     return end_date.strftime('%Y-%m-%d')



# def calculate_end_date(start_date_str, days=7):
#     """
#     Calculate the end date given a start date and the number of days to add.
    
#     This version of the function can handle start dates with or without time information.
    
#     Parameters:
#     - start_date_str (str): The start date in 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS' format.
#     - days (int): The number of days to add to the start date.
    
#     Returns:
#     - str: The end date in 'YYYY-MM-DD' format.
#     """
#     # Attempt to parse the start date string with time information
#     try:
#         start_date = datetime.strptime(start_date_str, '%Y-%m-%d %H:%M:%S')
#     except ValueError:
#         # If it fails, assume the string is in 'YYYY-MM-DD' format
#         start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
    
#     # Calculate the end date by adding the specified number of days to the start date
#     end_date = start_date + timedelta(days=days)
    
#     # Convert the end date back to a string in 'YYYY-MM-DD' format
#     return end_date.strftime('%Y-%m-%d')
# ---------------------------------------------------------------------------------------------------
# Plotting functions

def plot_int_stacked(df,height = 600, width = 1000):
  import plotly.graph_objects as go
  import pandas as pd
  import numpy as np

  # create the Plotly figure object
  fig = go.Figure(
    layout=go.Layout(
        height=600,  # Set the desired height of the plot in pixels
        width=800  # Set the desired width of the plot in pixels
    )
    )   

  # add traces to the figure object
  for column in df.columns:
      fig.add_trace(go.Scatter(x=df.index, y=df[column], mode='lines', stackgroup='one', fill='tonexty', name=column))

  # customize the plot layout
  fig.update_layout(title='Stacked Plot', xaxis_title='Time', yaxis_title='Power (MW)', showlegend=True,width=width, height=height)

  # display the plot
  fig.show()

def interactive_plot(df, height=600, width=1000, y_limits=None,title = None, yaxis_title = None,stacked = False,add_line = False, line_data = None):
    import plotly.graph_objects as go
    fig = go.Figure(layout=go.Layout(height=height, width=width))

    # Add traces to the figure object
    if stacked == True:
        for column in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df[column], mode='lines', stackgroup='one', fill='tonexty', name=column))

    if stacked == False:
        for column in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df[column], name=column))

    if add_line == True:
        for column in line_data.columns:
            fig.add_trace(go.Scatter(x=line_data.index, y=line_data[column], name=column))
    
    # Customize the plot layout
    fig.update_layout(
        title=title,
        xaxis_title='Time',
        yaxis_title=yaxis_title,
        autosize=False,
        width=width,
        height=height,
        margin=dict(l=50, r=50, b=100, t=100, pad=4)
    )
    # Set the y-limits if provided
    if y_limits:
        fig.update_yaxes(range=y_limits)

    # Display the plot
    fig.show()

# Colab interaction


def view_network(df):
  import ipywidgets as widgets
  from IPython.display import display, clear_output

  # get the names of all components in the network
  components = ['buses','generators','carriers','links','lines','loads','loads_t','generators_t.p_max_pu','snapshots','snapshot_weightings','investment_periods','investment_period_weightings']

  # create dropdown
  dropdown = widgets.Dropdown(
      options=components,
      description='Network Components:',
  )

  output = widgets.Output()

  # define the function to be called on change
  def on_change(change):
      if change['type'] == 'change' and change['name'] == 'value':
          with output:
              # clear previous selection display and print name of the component
              clear_output()
              # print("Selected component: %s" % change['new'])
              display(getattr(df, change['new']))

  dropdown.observe(on_change, names='value')

  display(dropdown)
  display(output)


def create_editable_table(df: pd.DataFrame, editable_columns):
    # Define the input fields
    fields = ['Index'] + editable_columns

    # Get the row names
    row_names = df.index.tolist()

    # Create a row with labels
    labels = [widgets.Label(value=field, layout=widgets.Layout(width='200px')) for field in fields]
    header = widgets.HBox(labels)

    # Create text fields and arrange them in HBox for each row
    rows = []
    for name in row_names:
        row = [widgets.Label(value=str(name), layout=widgets.Layout(width='200px'))]
        for field in editable_columns:
            row.append(widgets.Text(value=str(df.loc[name, field]), layout=widgets.Layout(width='200px')))
        rows.append(widgets.HBox(row))

    # Combine all rows in VBox
    table = widgets.VBox(rows)

    # Define the button
    update_button = widgets.Button(description='Update DataFrame')

    # Create an Output widget to capture and return the DataFrame
    output = widgets.Output()

    # Define the function to update the DataFrame
    def update_dataframe(button):
        with output:
            for i, name in enumerate(row_names):
                for j, field in enumerate(editable_columns):
                    try:
                        # Try to convert the value to a float
                        df.loc[name, field] = float(rows[i].children[j+1].value)  # Note the +1 here, because the first child is now the index label
                    except ValueError:
                        # If the value can't be converted to a float, leave it as a string
                        df.loc[name, field] = rows[i].children[j+1].value
            clear_output()  # Clear previous output
            # display(df)

    # Set the button's on_click event to update the DataFrame
    update_button.on_click(update_dataframe)

    # Display everything
    display(header, table, update_button, output)

def update_component(df):
    import ipywidgets as widgets
    from IPython.display import display, clear_output
    from math import ceil

    # get the names of all components in the network
    components = ['buses','generators','carriers','links','lines','loads','loads_t','stores',
                  'generators_t','snapshots','snapshot_weightings','investment_periods','investment_period_weightings']

    # create dropdown
    dropdown = widgets.Dropdown(
        options=components,
        description='Network Components:',
    )

    output = widgets.Output()

    # Create a dict to store the checkbox widgets
    checkbox_dict = {}

    #define the function to be called on change
    def on_change(change):
        if change['type'] == 'change' and change['name'] == 'value':
            # clear previous selection display
            with output:
                clear_output()
            
            # get the column names of the selected dataframe in a list
            column_list = getattr(df, change['new']).columns.tolist()
            
            # Create checkboxes
            checkbox_dict.clear()
            for col in column_list:
                checkbox_dict[col] = widgets.Checkbox(value=False, description=col, disabled=False)

            # Format checkboxes in three columns
            boxes_in_a_row = 3
            rows = ceil(len(column_list) / boxes_in_a_row)

            # Create a VBox for each row
            vbox_list = []
            for i in range(rows):
                hbox = widgets.HBox(list(checkbox_dict.values())[i*boxes_in_a_row:(i+1)*boxes_in_a_row])
                vbox_list.append(hbox)

            # Create a VBox that contains all rows
            vbox = widgets.VBox(vbox_list)

            # Create a button to confirm the selection and call the editable table function
            confirm_button = widgets.Button(description='Confirm Selection')

            def on_button_clicked(button):
                # Get the selected columns
                selected_columns = [col for col, checkbox in checkbox_dict.items() if checkbox.value]
                
                # Call the editable table function
                create_editable_table(getattr(df, dropdown.value), selected_columns)
                
            confirm_button.on_click(on_button_clicked)

            with output:
                display(vbox, confirm_button)

    dropdown.observe(on_change, names='value')

    display(dropdown, output)

def savenetwork(df):
  from google.colab import files
  # Create an input field for the filename
  filename_input = widgets.Text(value='my_network', description='Filename:', layout=widgets.Layout(width='300px'))

  # Create a button to save and download the file
  download_button = widgets.Button(description='Save and Download')

  # Define the function to save and download the file
  def save_and_download(button):
      filename = filename_input.value + '.h5'  # add the extension to the filename
      df.export_to_hdf5(filename)  # save the network to an HDF5 file
      files.download(filename)  # download the file

  # Set the button's on_click event to save and download the file
  download_button.on_click(save_and_download)

  # Display everything
  display(filename_input, download_button)

