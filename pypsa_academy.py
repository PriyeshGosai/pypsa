import os
import pandas as pd
from ipywidgets import interact, fixed
from IPython.display import display, clear_output
import ipywidgets as widgets
import logging
from concurrent.futures import ThreadPoolExecutor


def convert_sheet_to_csv(xls, sheet_name, csv_folder_path):
    df = xls.parse(sheet_name)
    csv_file_path = os.path.join(csv_folder_path, f"{sheet_name}.csv")
    df.to_csv(csv_file_path, index=False)
    logging.info(f"Converted {sheet_name} to CSV.")
    return csv_file_path

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
    components = {"buses", "carriers", "generators", "generators-p_max_pu",
    "generators-p_min_pu", "generators-p_set","line_types", "lines",
    "links", "links-p_max_pu","links-p_min_pu", "links-p_set",
    "loads", "loads-p_set", "shapes","shunt_impedances",
    "snapshots", "storage_units", "stores","sub_networks",
    "transformer_types","transformers"}
    created_csv_files = []

    # Ensure the CSV folder exists
    os.makedirs(csv_folder_path, exist_ok=True)

    # Clear only relevant CSV files in the folder
    for item in os.listdir(csv_folder_path):
        if item.endswith(".csv") and item.replace(".csv", "") in components:
            os.remove(os.path.join(csv_folder_path, item))

    try:
        xls = pd.ExcelFile(excel_file_path)
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(convert_sheet_to_csv, xls, sheet_name, csv_folder_path)
                       for sheet_name in xls.sheet_names if sheet_name in components]
            for future in futures:
                created_csv_files.append(future.result())
    except Exception as e:
        logging.error(f"Error converting Excel to CSV: {e}")
        return []
    finally:
        if xls is not None:
            xls.close()
            print('Excel file is closed')

    logging.info(f"Conversion complete. CSV files are saved in '{csv_folder_path}'")
    return csv_folder_path


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

def interactive_plot(df, height=600, width=1000, y_limits=None, title=None, yaxis_title=None, stacked=False, add_line=False, line_data=None):
    import plotly.graph_objects as go
    fig = go.Figure(layout=go.Layout(height=height, width=width))

    # Add traces to the figure object
    if stacked:
        for column in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df[column], mode='lines', stackgroup='one', fill='tonexty', name=column))
    else:
        for column in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df[column], name=column))

    if add_line:
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

    # Return the figure object
    return fig


# Colab interaction
def view_network(df):
  import ipywidgets as widgets
  from IPython.display import display, clear_output

  # get the names of all components in the network
  components = ['buses','generators','carriers','links','lines','loads','stores','snapshots']

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

