import click
from functools import partial
from pathlib import Path
import tifffile
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from sys_micro_pytools.preprocess.flat_field import get_flat_field_files
from sys_micro_pytools.df import link_df2plate_layout
from sys_micro_pytools.visualize import create_palette
from sys_micro_pytools.visualize.channel_plots import create_channel_plots
from sys_micro_pytools.visualize.grid_plots import get_df_images, create_grid_plot
from sys_micro_pytools.visualize.count_plots import create_count_df, create_count_plot

def empty_to_none(ctx, param, value):
    if value == ():
        return None
    return value

def validate_max_items(ctx, param, value, count):
    """
    Validates that a multiple option doesn't exceed a maximum count.
    
    Args:
        ctx: Click context
        param: The parameter being validated
        value: The value being validated (list from multiple=True)
        count: Number of required items
    """
    if not len(value) == count:
        raise click.BadParameter(
            f'{count} items required for {param.name}, got {len(value)}'
        )
    return value

@click.group()
def cli():
    pass


@cli.command(name='channel-plots')
@click.option( '-i', '--input_path', type=click.Path(exists=True), required=True,
              help='Path to directory containing images')
@click.option( '-o', '--output_path', type=click.Path(), required=True,
              help='Path to directory where output will be saved')
@click.option('--img_type', type=click.Choice(['multi_channel', 'grayscale']), default='grayscale',
              help='Type of image to plot. Options: multi_channel, grayscale')
@click.option('--channels2use', type=int, multiple=True, default=(0,),
              help='Channels to use for making the plots')
@click.option('--suffix', type=str, default='.nd2',
              help='Suffix of image files')
@click.option('--normalize', is_flag=True,
              help='Normalize the images')
@click.option('--ref_wells', type=str, multiple=True, callback=empty_to_none,
              help='Well(s) to use as reference for normalization using its percentiles')
@click.option('--filename_well_idx', type=int, multiple=True, default=(4,7), 
              callback=partial(validate_max_items, count=2),
              help='Start and stop indices of the well name in the filename (default: (4,7))')
@click.option('--filename_field_idx', type=int, multiple=True, default=(17,21),
              callback=partial(validate_max_items, count=2),
              help='Start and stop indices of the field number in the filename (default: (17,21))')
@click.option('--flat_field_path', type=click.Path(), default=None,
              help='Path to image to use as flat field correction')
@click.option('--percentiles', type=float, multiple=True, default=(0.1, 99.9),
              callback=partial(validate_max_items, count=2),
              help='Percentiles to use for image normalization (default: (0.1, 99.9))')
@click.option('--pattern2ignore', type=str, multiple=True, callback=empty_to_none,
              help='Pattern to ignore in filename')
@click.option('--patterns2have', type=str, multiple=True, callback=empty_to_none,
              help='Patterns to have in filename. If not None, only files with these patterns will be processed')
@click.option('--field_idx', type=int, default=None,
              help='Index of field to plot (default: None)')
@click.option('--output_type', type=str, multiple=True, default=('channels', 'composite'),
              help='Type of output to save (default: (channels, composite))')
def create_channel_plots_cli(input_path, output_path, img_type, channels2use, suffix, 
                             normalize, ref_wells, filename_well_idx, filename_field_idx, 
                             flat_field_path, percentiles, pattern2ignore, patterns2have, 
                             field_idx, output_type):
    """Main function that processes command line arguments and calls create_channel_plots."""

    if ref_wells is not None:
        if isinstance(ref_wells, str):
            ref_wells = (ref_wells,)
        else:
            ref_wells = tuple(str(well) for well in ref_wells)

    if pattern2ignore is not None:
        if isinstance(pattern2ignore, str):
            pattern2ignore = (pattern2ignore,)

    if patterns2have is not None:
        if isinstance(patterns2have, str):
            patterns2have = (patterns2have,)
    
    if output_type is not None:
        if isinstance(output_type, str):
            output_type = (output_type,)

    # Call the function
    create_channel_plots(
        input_path=input_path,
        output_path=output_path,
        img_type=img_type,
        channels2use=channels2use,
        suffix=suffix,
        normalize=normalize,
        ref_wells=ref_wells,
        filename_well_idx=filename_well_idx,
        filename_field_idx=filename_field_idx,
        flat_field_path=flat_field_path,
        percentiles=percentiles,
        pattern2ignore=pattern2ignore,
        patterns2have=patterns2have,
        field_idx=field_idx,
        output_type=output_type
    )

@cli.command(name='grid-plot')
@click.option( '-i', '--input_path', type=click.Path(exists=True), required=True,
              help='Path to directory containing images')
@click.option( '-o', '--output_path', type=click.Path(), required=True,
              help='Path to directory where output will be saved')
@click.option('--plate_layout', type=click.Path(), required=True,
              help='Path to plate layout file')
@click.option('--suffix', type=str, default='.nd2',
              help='Suffix of image files')
@click.option('--filename_well_idx', type=int, multiple=True, default=(4,7), 
              callback=partial(validate_max_items, count=2),
              help='Start and stop indices of the well name in the filename (default: (4,7))')
@click.option('--filename_field_idx', type=int, multiple=True, default=(17,21),
              callback=partial(validate_max_items, count=2),
              help='Start and stop indices of the field number in the filename (default: (17,21))')
@click.option('--skip_wells', type=str, multiple=True, default=[],
              help='List of wells to skip')
@click.option('--img_type', type=click.Choice(['multi_channel', 'grayscale', 'mask']), default='grayscale',
              help='Type of image to plot. Options: multi_channel, grayscale, mask')
@click.option('--channels2use', type=int, multiple=True, default=(0,),
              help='Channels to use for making the plots')
@click.option('--ref_wells', type=str, multiple=True, callback=empty_to_none,
            help='Well(s) to use as reference for normalization using its percentiles')
@click.option('--masks', is_flag=True,
              help='Indicate that image is a mask. FF correction will not be applied')
@click.option('--flat_field_path', type=click.Path(), default=None,
              help='Path to image to use as flat field correction')
@click.option('--cmap', type=str, default='cet_glasbey',
              help='Color map to use image borders in grid plot')
@click.option('--condition_vars', type=str, multiple=True, default=('Treat', 'Dose'),
              help='Variables to use for color encoding')
@click.option('--conditions2remove', type=str, multiple=True, callback=empty_to_none,
              help='List of condition combinations to remove from the palette, e.g. --conditions2remove "DMSO,L"')
@click.option('--check_batches', is_flag=True,
              help='Check if input_path contains subdirectories')
@click.option('--field_idx', type=int, multiple=True, default=None,
              help='Index of the field to plot for each well. If None, a random field will be selected')
@click.option('--plate_id', type=str, default=None,
              help=('Plate ID to use. Overwrites plate_layout automatically found plate ID to use. '
                    'This is useful if the plate ID is not in the filename.'))
@click.option('--rep_id', type=str, default=None,
              help=('Rep ID to use. Overwrites plate_layout automatically found rep ID to use. '
                    'This is useful if the rep ID is not in the filename.'))
def create_grid_plot_cli(input_path, output_path, plate_layout, suffix, filename_well_idx, 
                         filename_field_idx, skip_wells, img_type, channels2use, ref_wells, 
                         masks, flat_field_path, cmap, condition_vars, conditions2remove, 
                         check_batches, field_idx, plate_id, rep_id):
    """Main function that processes command line arguments and calls create_grid_plot."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    if isinstance(skip_wells, str):
        skip_wells = [skip_wells]

    if isinstance(condition_vars, str):
        condition_vars = [condition_vars]
    
    if conditions2remove is not None:
        conditions2remove = [tuple(condition.split(',')) for condition in conditions2remove]

    if isinstance(field_idx, int):
        field_idx = [field_idx]
    for idx in field_idx:
        assert idx >= 0, 'field_idx must be a positive integer'

    # Set color palette for treatments
    plate_layout = pd.read_csv(plate_layout, sep=",|;", engine='python'); 
    plate_layout.columns = [col.capitalize() for col in plate_layout.columns]
    plate_layout['Plate'] = plate_layout['Plate'].astype(str)
    plate_layout['Well'] = plate_layout['Well'].astype(str)
    plate_layout['Row'] = plate_layout['Row'].astype(str)
    plate_layout['Col'] = plate_layout['Col'].astype(int)

    # Set color palette
    palette, _ = create_palette(
        plate_layout,
        condition_vars=condition_vars,
        cmap=cmap,
        conditions2remove=conditions2remove,
    )

    # Get flat field images
    if flat_field_path is not None:
        flat_field_files = get_flat_field_files(flat_field_path)
    else:
        flat_field_files = None
    
    # Image path
    input_path = Path(input_path)
    df_images = get_df_images(
        input_path,
        check_batches,
        suffix,
        filename_well_idx,
        filename_field_idx,
        skip_wells,
        plate_id,
        rep_id
    )
    df_images.columns = [col.capitalize() for col in df_images.columns]
    df_images['Plate'] = df_images['Plate'].astype(str)
    df_images['Well'] = df_images['Well'].astype(str)
    df_images['Row'] = df_images['Row'].astype(str)
    df_images['Col'] = df_images['Col'].astype(int)
    df_images['Field'] = df_images['Field'].astype(int)

    # Merge plate layout with df_images
    df_images = link_df2plate_layout(df_images, plate_layout)

    # Plot a single image per row/col combination
    for dir in tqdm(sorted(df_images['Dir'].unique())):
        # Load flat field
        if flat_field_files is not None:
            flat_field_file = [f for f in flat_field_files if Path(dir).stem in f.stem and 'FF' in f.stem][0]
            flat_field = tifffile.imread(str(flat_field_file)).astype(float)
        else:
            flat_field = None

        df_dir = df_images[df_images['Dir'] == dir]

        fig = create_grid_plot(
            df_dir,
            flat_field,
            condition_vars,
            palette,
            field_idx=field_idx,
            img_type=img_type,
            channels2use=channels2use,
            ref_wells=ref_wells,
            title=dir
        )

        # Save figure
        filename = Path(output_path).joinpath(f'{dir}.jpg')
        filename.parent.mkdir(exist_ok=True, parents=True)
        print(f'Saving figure to {filename}')
        fig.savefig(filename, dpi=600)
        plt.close()


@cli.command(name='count-plot')
@click.option( '-i', '--input_path', type=click.Path(exists=True), required=True,
              help='Path to directory containing images')
@click.option( '-o', '--output_path', type=click.Path(), required=True,
              help='Path to directory where output will be saved')
@click.option('--plate_layout', type=click.Path(), required=True,
              help='Path to plate layout file')
@click.option('--suffix', type=str, default='.tif',
              help='Suffix of image files')
@click.option('--filename_well_idx', type=int, multiple=True, default=(4,7), 
              callback=partial(validate_max_items, count=2),
              help='Start and stop indices of the well name in the filename (default: (4,7))')
@click.option('--filename_field_idx', type=int, multiple=True, default=(17,21),
              callback=partial(validate_max_items, count=2),
              help='Start and stop indices of the field number in the filename (default: (17,21))')
@click.option('--skip_wells', type=str, multiple=True, default=[],
              help='List of wells to skip')
@click.option('--cmap', type=str, default='cet_glasbey',
              help='Color map to use for boxplots')
@click.option('--condition_vars', type=str, multiple=True, default=('Treat', 'Dose'),
              help='Variables to use for color encoding')
@click.option('--conditions2remove', type=str, multiple=True, callback=empty_to_none,
              help='List of condition combinations to remove from the palette, e.g. --conditions2remove "DMSO,L"')
@click.option('--check_batches', is_flag=True,
              help='Check if input_path contains subdirectories')
@click.option('--y_label', type=str, default='Object Count',
              help='Label for y-axis')
@click.option('--y_lim', type=float, multiple=True, callback=empty_to_none,
              help='Limits for y-axis (min, max)')
@click.option('--box_width', type=float, default=0.8,
              help='Width of the boxes in the boxplot')
@click.option('--jitter', is_flag=True,
              help='Whether to add jitter to the points')
@click.option('--plot_type', type=click.Choice(['box', 'violin', 'swarm']),
              help='Type of plot to create')
@click.option('--output_format', type=str, default='png',
              help='Format for output figures')
@click.option('--dpi', type=int, default=150,
              help='DPI for output figures')
@click.option('--save_csv', is_flag=True,
              help='Whether to save counts to CSV file')
@click.option('--plate_id', type=str, default=None,
              help=('Plate ID to use. Overwrites plate_layout automatically found plate ID to use. '
                    'This is useful if the plate ID is not in the filename.'))
@click.option('--rep_id', type=str, default=None,
              help=('Rep ID to use. Overwrites plate_layout automatically found rep ID to use. '
                    'This is useful if the rep ID is not in the filename.'))
def create_count_plot_cli(input_path, output_path, plate_layout, suffix, filename_well_idx, 
                         filename_field_idx, skip_wells, cmap, condition_vars, conditions2remove, 
                         check_batches, y_label, y_lim, box_width, jitter, plot_type, output_format, 
                         dpi, save_csv, plate_id, rep_id):
    """Main function that processes command line arguments and calls create_count_plot."""

    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    if isinstance(skip_wells, str):
        skip_wells = [skip_wells]

    if isinstance(condition_vars, str):
        condition_vars = [condition_vars]
    
    if conditions2remove is not None:
        conditions2remove = [tuple(condition.split(',')) for condition in conditions2remove]

    # Set color palette for treatments
    plate_layout = pd.read_csv(plate_layout, sep=",|;", engine='python'); 
    plate_layout.columns = [col.capitalize() for col in plate_layout.columns]
    plate_layout['Plate'] = plate_layout['Plate'].astype(str)
    plate_layout['Well'] = plate_layout['Well'].astype(str)
    plate_layout['Row'] = plate_layout['Row'].astype(str)
    plate_layout['Col'] = plate_layout['Col'].astype(int)

    # Set color palette
    palette, _ = create_palette(
        plate_layout,
        condition_vars=condition_vars,
        cmap=cmap,
        conditions2remove=conditions2remove,
    )
    
    # Image path
    input_path = Path(input_path)
    df_images = get_df_images(
        input_path,
        check_batches,
        suffix,
        filename_well_idx,
        filename_field_idx,
        skip_wells,
        plate_id,
        rep_id
    )
    df_images.columns = [col.capitalize() for col in df_images.columns]
    df_images['Plate'] = df_images['Plate'].astype(str)
    df_images['Well'] = df_images['Well'].astype(str)
    df_images['Row'] = df_images['Row'].astype(str)
    df_images['Col'] = df_images['Col'].astype(int)
    df_images['Field'] = df_images['Field'].astype(int)

    # Merge plate layout with df_images
    df_images = link_df2plate_layout(df_images, plate_layout)
    
    # Count objects in masks
    df_counts = create_count_df(df_images)
    
    # For each directory, create a count boxplot
    for dir in tqdm(sorted(df_counts['Dir'].unique()), desc="Creating plots"):
        df_dir = df_counts[df_counts['Dir'] == dir]
        
        # Create boxplot
        fig = create_count_plot(
            df_dir,
            condition_vars,
            palette,
            title=f"{dir} - Object Counts",
            y_label=y_label,
            y_lim=y_lim,
            box_width=box_width,
            jitter=jitter,
            plot_type=plot_type
        )
        
        # Save figure
        filename = output_path.joinpath(f'{dir}_count_plot.{output_format}')
        fig.savefig(
            filename,
            dpi=dpi,
            bbox_inches='tight'
            )
        plt.close(fig)
        print(f'Saved count plot to {filename}')
        
        # Save counts to CSV
        if save_csv:
            csv_filename = output_path.joinpath(f'{dir}_counts.csv')
            df_dir.to_csv(csv_filename, index=False)
            print(f'Saved counts to {csv_filename}')

if __name__ == '__main__':
    cli()