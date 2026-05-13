import click
from pathlib import Path
from string import ascii_uppercase

from sys_micro_pytools.cli_utils import split_ws

from .plate_grid2table import plate_grid2table, plot_layout

PLATE_TYPE_CATEGORIES = {
    96: (list(ascii_uppercase[:8]), list(range(1, 13))),
    384: (list(ascii_uppercase[:16]), list(range(1, 25))),
}

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
@click.option('--plate_type', type=click.Choice(['96', '384']), default='96',
              help="Plate format used to derive default row/col categories. Ignored when --row_categories or --col_categories are provided.")
@click.option('--row_categories', type=str, default=None, callback=split_ws(item_type=str),
              help=('Categories for the row variable. Overrides the defaults implied by --plate_type. '
                    'e.g. --row_categories "A B C D E F G H"'))
@click.option('--col_categories', type=str, default=None, callback=split_ws(item_type=int),
              help=('Categories for the column variable. Overrides the defaults implied by --plate_type. '
                    'e.g. --col_categories "1 2 3 4 5 6 7 8 9 10 11 12"'))
@click.option('--var_order', type=str, default=None, callback=split_ws(item_type=str),
              help='Order of the variables in the visualization, e.g. --var_order "Treat Dose Well"')
@click.option('--ncols', type=click.INT, default=3,
              help="Number of columns for the subplots in the visualization.")
@click.option('--add_annot', is_flag=True,
              help="Whether to add annotations to the heatmap.")
@click.option('--numeric_vars', type=str, default=None, callback=split_ws(item_type=str),
              help='Variables to be treated as numeric in the visualization, e.g. --numeric_vars "Conc Vol"')
@click.option('--remove_rows_with_na', is_flag=True,
              help="Whether to remove rows with NA values from the dataframe.")
def plate_grid2table_cli(input_path, output_path, filename, visualize, plate_id,
                         plate_type, row_categories, col_categories, var_order, ncols,
                         add_annot, numeric_vars, remove_rows_with_na):
    """Convert a plate layout provided as grids in a .xlsx file(s) into a table in a .csv file.

    INPUT_PATH: Path to the CSV or Excel file(s) containing the plate layouts.
    OUTPUT_PATH: Path to save the merged layout dataframe.
    """
    # Merge the plate layouts into a single dataframe
    df = plate_grid2table(input_path, remove_rows_with_na)

    # Save the merged layout dataframe
    input_path_obj = Path(input_path)
    output_path = Path(output_path) if output_path else input_path_obj.parent
    filename = Path(filename) if filename else Path(input_path_obj.name).with_suffix('.csv')
    assert filename.suffix == '.csv', "Output file must be a CSV file."
    output_file = output_path / filename
    if output_file.resolve() == input_path_obj.resolve():
        raise click.UsageError(
            f"Output file '{output_file}' matches the input file. "
            "Choose a different --output_path or --filename to avoid overwriting the input."
        )
    df.to_csv(output_file, index=False)

    # Visualize the plate layout
    if visualize:
        default_rows, default_cols = PLATE_TYPE_CATEGORIES[int(plate_type)]
        if row_categories is None:
            row_categories = default_rows
        if col_categories is None:
            col_categories = default_cols
        if plate_id is None:
            plate_id = sorted(df['plate'].unique())
        for plate in plate_id:
            plot_layout(df, plate, row_categories, col_categories, var_order, 
                        ncols, add_annot, numeric_vars)

if __name__ == '__main__':
    df()