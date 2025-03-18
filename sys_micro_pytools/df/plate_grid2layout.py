from argparse import ArgumentParser
from pathlib import Path
import re
from typing import List, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns


def plate_grid2layout(paths: Union[List[Union[str, Path]], Union[str, Path]]) -> pd.DataFrame:
    ''' Merge multiple well plate layouts into a single dataframe, considering multiple files or multiple sheets.
        
        Each layout is a N x M matrix in which the first row is the header (column names)
        and the first column is the row names. The cell in the first row and first column 
        mentions the variable name. The rest of the cells, from position [1,1] to [N,M], 
        contain the values of the variable in the well plate.

        Between each layout, there is at least one blank row.

        Each row in the final dataframe corresponds to a well in the well plate and each column 
        corresponds to a variable, with an additional 'plate' column specifying the plate index.
        The plate index is extracted from either the filenames or the sheet name using the pattern "_plateX"
        where X is an alphanumeric character or sequence of such characters.
    '''
    if isinstance(paths, (str, Path)):
        paths = [paths]  # Ensure paths is always a list

    layout_dfs = []
    plate_pattern = r"plate([A-Za-z0-9]+)"  # Regex to match alphanumeric characters after "plate"

    for path in paths:
        path = Path(path)
        if Path(path).suffix in ['.xlsx', '.xls']:
            # Process as an Excel file possibly containing multiple sheets
            xls = pd.ExcelFile(path)
            sheets_names = xls.sheet_names
            for sheet in sheets_names:
                df = pd.read_excel(xls, sheet_name=sheet, header=None)
                # Extract plate index from the sheet name if it exists, else from the filename
                if re.search(plate_pattern, sheet, re.IGNORECASE):
                    plate_index = extract_plate_index(sheet, plate_pattern)
                elif len(sheets_names) == 1:
                    plate_index = extract_plate_index(str(path), plate_pattern)
                layout_df = process_layout(df, plate_index)
                layout_dfs.append(layout_df)
        else:
            # Process as a single CSV file or similar
            plate_index = extract_plate_index(str(path), plate_pattern)
            df = pd.read_csv(path, header=None)
            layout_df = process_layout(df, plate_index)
            layout_dfs.append(layout_df)

    # Concatenate all individual layout dataframes
    combined_df = pd.concat(layout_dfs, ignore_index=True)

    # Pivot the combined dataframe to wide format with unique identifiers for each well and plate
    final_df = combined_df.pivot_table(
        index=['plate', 'row', 'col','well'],
        columns='variable',
        values='value',
        aggfunc='first'
        ).reset_index()
    final_df.columns.name = None  # Remove the hierarchy in the columns

    return final_df

def extract_plate_index(name: str, pattern: str) -> str:
    """ Extracts the plate index from a filename or sheet name using a regex pattern. """
    match = re.search(pattern, name, re.IGNORECASE)
    return match.group(1) if match else 'unknown'

def process_layout(df, plate_index):
    """ Processes individual layouts from a DataFrame. """
    start = 0
    layout_dfs = []
    while start < len(df):
        if df.iloc[start].isnull().all():
            start += 1
            continue

        # Determine end
        end = start
        while end < len(df) and not df.iloc[end].isnull().all():
            end += 1

        current_layout = df.iloc[start:end]
        variable_name = current_layout.iloc[0, 0]  # Variable name
        headers = pd.to_numeric(current_layout.iloc[0, 1:]).astype('Int64').values
        row_labels = current_layout.iloc[1:, 0].values
        data = current_layout.iloc[1:, 1:].values
        layout_df = pd.DataFrame(data, columns=headers, index=row_labels)
        melted_df = layout_df.reset_index().melt(id_vars='index')
        melted_df.columns = ['row', 'col', 'value']
        melted_df['variable'] = variable_name
        melted_df['well'] = melted_df['row'] + melted_df['col'].astype(str).str.zfill(2)
        melted_df['plate'] = plate_index
        layout_dfs.append(melted_df)
        start = end + 1
    return pd.concat(layout_dfs, ignore_index=True)

def plot_layout(df, selected_plate, row_categories=None, col_categories=None, var_order=None, ncols=3, add_annot=False):
    """
    Creates a plot of the plate layout for a specified plate with each variable as a separate subplot.
    Assumes df has columns 'plate', 'well', and other columns representing variables.
    'well' column should be in the format A01, A02, ..., H12.
    
    Parameters:
    df (DataFrame): The DataFrame containing the well and variable data.
    selected_plate (str or int): The plate index to be visualized.
    """

    # Filter the dataframe for the selected plate
    df = df[df['plate'] == selected_plate].copy()

    # Set row and column as categorical for proper ordering in the heatmap
    row_categories = row_categories if row_categories else sorted(df['row'].unique())
    col_categories = col_categories if col_categories else sorted(df['col'].unique())
    df['row'] = pd.Categorical(df['row'], categories = row_categories, ordered=True)
    df['col'] = pd.Categorical(df['col'], categories = col_categories, ordered=True)

    # Extract unique variables
    factors = ['plate', 'well', 'row', 'col']
    variables = [col for col in df.columns if col not in factors] if var_order is None else var_order
    assert all(var in df.columns for var in variables), "One or more variables not found in the dataframe."

    # Drop all other columns
    df = df[factors + variables].copy()

    n_vars = len(variables)
    vars2int = {}
    for var in variables:
        df[var] = pd.Categorical(df[var], categories=df[var].unique(), ordered=True)
        vars2int[var] = {val: i for i, val in enumerate(df[var].cat.categories)}

    # Determine layout of subplots
    nrows = (n_vars + ncols - 1) // ncols
    
    # Determine the layout of the subplots, including space for color bars
    _, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 5, nrows * 4))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.flatten()

    for i, var in enumerate(variables):
        if i >= len(axes):
            break
        df_pivot = df.pivot_table(
            values=var,
            index='row',
            columns='col',
            aggfunc='first',
            observed=False,
            dropna=False
            )
        df_pivot = df_pivot.map(lambda x: vars2int[var][x] if x in vars2int[var] else x).astype(float)
        ax = axes[i]
        
        num_categories = len(vars2int[var])
        palette = sns.color_palette("Set3", n_colors=num_categories)
        ticklabels = list(vars2int[var].keys())
        ticks = np.arange(num_categories)

        if num_categories == 1:
            # Create a ListedColormap with the first color
            cmap = ListedColormap([palette[0]])
            boundaries = [-0.5, 0.5]
        else:
            cmap = ListedColormap(palette)
            boundaries = np.arange(num_categories + 1) - 0.5

        # Create heatmap
        int2vars = {v: k for k, v in vars2int[var].items()}
        annot = df_pivot.map(lambda x: str(int2vars[x]) if x in int2vars else x)

        sns.heatmap(
            df_pivot,
            ax=ax,
            cmap=cmap,
            annot=annot if add_annot else None,
            fmt='s',
            square=True,
            cbar=True,
            cbar_kws={"ticks": ticks, "boundaries": boundaries},
            annot_kws={"size": 8},
            linewidths=0.5, linecolor='black'
            )
        cbar = ax.collections[0].colorbar
        cbar.set_ticklabels(ticklabels)
        ax.set_xlim(-0.5, len(col_categories) + 0.5)
        ax.set_ylim(-0.5, len(row_categories) + 0.5)
        ax.invert_yaxis()
        ax.set_yticklabels(row_categories,rotation=0)
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")
        ax.set_title(f"{var}")

    # Turn off any unused axes
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.suptitle(f"Plate {selected_plate} Layout")
    plt.tight_layout()
    plt.show()


def parse_args():
    parser = ArgumentParser(description=("Convert a plate layout as a list in a .csv file from a plate layout provided in a .xlsx file(s)."))
    parser.add_argument('-i', '--input_path', type=str, required=True, nargs='+',
                        help="Path to the CSV or Excel file(s) containing the plate layouts.")
    parser.add_argument('-o', '--output_path', type=str, required=False, default=None,
                        help="Path to save the merged layout dataframe.")
    parser.add_argument('-f', '--filename', type=str, default=None,
                        help=("The filename of the plate layout. If not specified, input_path will be used."
                              "For the latter, if input_path is a list of files and "
                              "there is a shared parent folder, the parent folder will be used."
                              "If input_path is a list of files and there is no shared parent folder, "
                              "the first file will be used.")
                              )
    parser.add_argument('-v', '--visualize', action='store_true',
                        help="Whether to visualize the plate layout.")
    parser.add_argument('-p', '--plate_id', type=str, default=None,
                        help="The plate index to be visualized. If not specified, all plates will be visualized.")
    parser.add_argument('--row_categories', type=str, nargs='+', default=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'],
                        help="Categories for the row variable.")
    parser.add_argument('--col_categories', type=int, nargs='+', default=list(range(1, 13)),
                        help="Categories for the column variable.")
    parser.add_argument('--var_order', type=str, nargs='*', default=None,
                        help="Order of the variables in the visualization.")
    parser.add_argument('--ncols', type=int, default=3,
                         help="Number of columns for the subplots in the visualization.")
    parser.add_argument('--add_annot', action='store_true',
                        help="Whether to add annotations to the heatmap.")
    args = parser.parse_args()

    if isinstance(args.plate_id, str):
        args.plate_id = [args.plate_id]

    return args

def main(args):
    # Merge the plate layouts into a single dataframe
    df = plate_grid2layout(args.input_path)

    # Save the merged layout dataframe
    if args.output_path:
        output_path = Path(args.output_path)  
        args.filename = Path(args.filename) if args.filename else None
        if args.filename is None:
            parent_folder = None
            if isinstance(args.input_path, list) and len(args.input_path) > 1:
                parent_folder = Path(args.input_path[0]).parent
                for path in args.input_path[1:]:
                    if Path(path).parent != parent_folder:
                        parent_folder = None
                        break
                if parent_folder:
                    args.filename = Path(parent_folder.name + '.csv')
        filename = Path(Path(args.input_path[0]).name).with_suffix('.csv') if args.filename is None else args.filename
        assert filename.suffix == '.csv', "Output file must be a CSV file."
        filename = output_path.joinpath(filename)
        df.to_csv(filename, index=False)

    # Visualize the plate layout
    if args.visualize:
        if args.plate_id is None:
            args.plate_id = sorted(df['plate'].unique())
        for plate in args.plate_id:
            plot_layout(df, plate, args.row_categories, args.col_categories, args.var_order, args.ncols, args.add_annot)

if __name__ == '__main__':
    args = parse_args()
    main(args)