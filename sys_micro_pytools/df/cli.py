import click
from pathlib import Path
from .plate_grid2table import plate_grid2table, plot_layout

def empty_to_none(ctx, param, value):
    if value == ():
        return None
    return value

@click.group()
def df():
    """DataFrame operations."""
    pass

@df.command(name='plate-grid2table')
@click.argument('input_path', type=click.Path(exists=True))
@click.option('-o', '--output_path', type=click.Path(exists=True), default=None,
              help=("Path to save the merged layout dataframe. If not specified, the output will be saved in the same directory as the input file(s)."))
@click.option('-f', '--filename', type=click.Path(), default=None,
              help="Name of the output file.")
@click.option('-v', '--visualize', is_flag=True,
              help="Whether to visualize the plate layout.")
@click.option('--plate_id', type=click.STRING, default=None,
              help="The plate index to be visualized. If not specified, all plates will be visualized.")
@click.option('--row_categories', type=click.STRING, multiple=True, default=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'],
              help="Categories for the row variable.")
@click.option('--col_categories', type=click.INT, multiple=True, default=list(range(1, 13)),
              help="Categories for the column variable.")
@click.option('--var_order', type=click.STRING, multiple=True, callback=empty_to_none,
              help="Order of the variables in the visualization.")
@click.option('--ncols', type=click.INT, default=3,
              help="Number of columns for the subplots in the visualization.")
@click.option('--add_annot', is_flag=True,
              help="Whether to add annotations to the heatmap.")
@click.option('--numeric_vars', type=click.STRING, multiple=True, callback=empty_to_none,
              help="Variables to be treated as numeric in the visualization.")
@click.option('--remove_rows_with_na', is_flag=True,
              help="Whether to remove rows with NA values from the dataframe.")
def plate_grid2table_cli(input_path, output_path, filename, visualize, plate_id, 
                         row_categories, col_categories, var_order, ncols, 
                         add_annot, numeric_vars, remove_rows_with_na):
    """Convert a plate layout provided as grids in a .xlsx file(s) into a table in a .csv file.

    INPUT_PATH: Path to the CSV or Excel file(s) containing the plate layouts.
    OUTPUT_PATH: Path to save the merged layout dataframe.
    """
    # Merge the plate layouts into a single dataframe
    df = plate_grid2table(input_path, remove_rows_with_na)

    # Save the merged layout dataframe
    if output_path:
        output_path = Path(output_path)  
        filename = Path(filename) if filename else None
        if filename is None:
            parent_folder = None
            if isinstance(input_path, list) and len(input_path) > 1:
                parent_folder = Path(input_path[0]).parent
                for path in input_path[1:]:
                    if Path(path).parent != parent_folder:
                        parent_folder = None
                        break
                if parent_folder:
                    filename = Path(parent_folder.name + '.csv')
        filename = Path(Path(input_path[0]).name).with_suffix('.csv') if filename is None else filename
        assert filename.suffix == '.csv', "Output file must be a CSV file."
        filename = output_path.joinpath(filename)
        df.to_csv(filename, index=False)

    # Visualize the plate layout
    if visualize:
        if plate_id is None:
            plate_id = sorted(df['plate'].unique())
        for plate in plate_id:
            plot_layout(df, plate, row_categories, col_categories, var_order, 
                        ncols, add_annot, numeric_vars)

if __name__ == '__main__':
    df()