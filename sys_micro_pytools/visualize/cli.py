import click
from pathlib import Path
import tifffile
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from sys_micro_pytools.cli_utils import split_ws
from sys_micro_pytools.preprocess.flat_field import get_flat_field_files
from sys_micro_pytools.df import link_df2plate_layout
from sys_micro_pytools.visualize import create_palette
from sys_micro_pytools.visualize.channel_plots import create_channel_plots
from sys_micro_pytools.visualize.grid_plots import get_df_images, create_grid_plot
from sys_micro_pytools.visualize.count_plots import create_count_df, create_count_plot


@click.group()
def cli():
    pass


@cli.command(name='channel-plots')
@click.option( '-i', '--input_path', type=click.Path(exists=True), required=True,
              help='Path to directory containing images')
@click.option( '-o', '--output_path', type=click.Path(), required=True,
              help='Path to directory where output will be saved')
@click.option('--img_type', type=click.Choice(['multichannel', 'grayscale']), default='grayscale',
              help='Type of image to plot. Options: multichannel, grayscale')
@click.option('--channels2use', type=str, default='0', callback=split_ws(item_type=int),
              help='Channel indices for plots, e.g. --channels2use "0 1 2"')
@click.option('--suffix', type=str, default='.nd2',
              help='Suffix of image files')
@click.option('--normalize_mode', type=click.Choice(['off', 'per_image', 'ref_wells', 'global']),
              default='per_image',
              help=('Intensity scaling: off, per_image, ref_wells (needs --ref_wells), '
                    'or global. Multichannel composites always use global.'))
@click.option('--ref_wells', type=str, default=None, callback=split_ws(item_type=str),
              help='Reference wells for normalization percentiles, e.g. --ref_wells "A01 A02"')
@click.option('--filename_well_idx', type=str, default='4 7',
              callback=split_ws(item_type=int, expected_count=2),
              help='Well-name slice start/stop indices in filename, e.g. --filename_well_idx "4 7"')
@click.option('--filename_field_idx', type=str, default='17 21',
              callback=split_ws(item_type=int, expected_count=2),
              help='Field slice start/stop indices in filename, e.g. --filename_field_idx "17 21"')
@click.option('--flat_field_path', type=click.Path(), default=None,
              help='Path to image to use as flat field correction')
@click.option('--percentiles', type=str, default='0.1 99.9',
              callback=split_ws(item_type=float, expected_count=2),
              help='Normalization percentiles low high, e.g. --percentiles "0.1 99.9"')
@click.option('--pattern2ignore', type=str, default=None, callback=split_ws(item_type=str),
              help='Substring(s) to ignore in filename(s), e.g. --pattern2ignore "junk tmp"')
@click.option('--patterns2have', type=str, default=None, callback=split_ws(item_type=str),
              help=('Only process filenames containing one of these substrings(s), '
                    'e.g. --patterns2have "GFP DAPI"'))
@click.option('--field_idx', type=int, default=None,
              help='Index of field to plot (default: None)')
@click.option('--output_type', type=str, default='channels composite',
              callback=split_ws(item_type=str),
              help='Outputs to save, e.g. --output_type "channels composite"')
@click.option('--colors', type=str, default=None, callback=split_ws(item_type=str),
              help=('Colors to use for each channel in the composite image, '
                    'e.g. --colors "cyan green magenta". If omitted, defaults of '
                    'create_composite are used.'))
def create_channel_plots_cli(input_path, output_path, img_type, channels2use, suffix,
                             normalize_mode, ref_wells, filename_well_idx, filename_field_idx,
                             flat_field_path, percentiles, pattern2ignore, patterns2have,
                             field_idx, output_type, colors):
    """Main function that processes command line arguments and calls create_channel_plots."""

    if ref_wells is not None:
        ref_wells = tuple(str(well) for well in ref_wells)

    # Call the function
    create_channel_plots(
        input_path=input_path,
        output_path=output_path,
        img_type=img_type,
        channels2use=channels2use,
        suffix=suffix,
        normalize_mode=normalize_mode,
        ref_wells=ref_wells,
        filename_well_idx=filename_well_idx,
        filename_field_idx=filename_field_idx,
        flat_field_path=flat_field_path,
        percentiles=percentiles,
        pattern2ignore=pattern2ignore,
        patterns2have=patterns2have,
        field_idx=field_idx,
        output_type=output_type,
        colors=list(colors) if colors is not None else None,
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
@click.option('--filename_well_idx', type=str, default='4 7',
              callback=split_ws(item_type=int, expected_count=2),
              help='Well-name slice start/stop indices in filename, e.g. --filename_well_idx "4 7"')
@click.option('--filename_field_idx', type=str, default='17 21',
              callback=split_ws(item_type=int, expected_count=2),
              help='Field slice start/stop indices in filename, e.g. --filename_field_idx "17 21"')
@click.option('--skip_wells', type=str, default=None, callback=split_ws(item_type=str),
              help='Wells to skip, e.g. --skip_wells "A01 H12"')
@click.option('--img_type', type=click.Choice(['multichannel', 'grayscale', 'mask']), default='grayscale',
              help='Type of image to plot. Options: multichannel, grayscale, mask')
@click.option('--channels2use', type=str, default='0', callback=split_ws(item_type=int),
              help='Channel indices for plots, e.g. --channels2use "0 1 2"')
@click.option('--normalize_mode', type=click.Choice(['off', 'per_image', 'ref_wells', 'global']),
              default='per_image',
              help=('Intensity scaling: off, per_image, ref_wells (needs --ref_wells), '
                    'or global. Multichannel grids always use global.'))
@click.option('--ref_wells', type=str, default=None, callback=split_ws(item_type=str),
              help='Reference wells for normalization percentiles, e.g. --ref_wells "A01 A02"')
@click.option('--percentiles', type=str, default='0.1 99.9',
              callback=split_ws(item_type=float, expected_count=2),
              help='Normalization percentiles low high, e.g. --percentiles "0.1 99.9"')
@click.option('--masks', is_flag=True,
              help='Indicate that image is a mask. FF correction will not be applied')
@click.option('--flat_field_path', type=click.Path(), default=None,
              help='Path to image to use as flat field correction')
@click.option('--cmap', type=str, default='cet_glasbey',
              help='Color map to use image borders in grid plot')
@click.option('--condition_vars', type=str, default='Treat Dose', callback=split_ws(item_type=str),
              help='Variables for palette conditions, e.g. --condition_vars "Treat Dose"')
@click.option('--conditions2remove', type=str, default=None, callback=split_ws(item_type=str),
              help=('Condition tuples to drop from palette; each token is comma-joined vars, '
                    'e.g. --conditions2remove "DMSO,L OtherTreat,H"'))
@click.option('--check_batches', is_flag=True,
              help='Check if input_path contains subdirectories')
@click.option('--field_idx', type=str, default=None, callback=split_ws(item_type=int),
              help=('Field index(es) per layout: one int for all wells, or one per grid cell, '
                    'e.g. --field_idx "3" or --field_idx "0 1 2 3". If omitted, random field per well.'))
@click.option('--plate_id', type=str, default=None,
              help=('Plate ID to use. Overwrites plate_layout automatically found plate ID to use. '
                    'This is useful if the plate ID is not in the filename.'))
@click.option('--rep_id', type=str, default=None,
              help=('Rep ID to use. Overwrites plate_layout automatically found rep ID to use. '
                    'This is useful if the rep ID is not in the filename.'))
def create_grid_plot_cli(input_path, output_path, plate_layout, suffix, filename_well_idx,
                         filename_field_idx, skip_wells, img_type, channels2use, normalize_mode,
                         ref_wells, percentiles, masks, flat_field_path, cmap, condition_vars,
                         conditions2remove, check_batches, field_idx, plate_id, rep_id):
    """Main function that processes command line arguments and calls create_grid_plot."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    skip_wells = skip_wells if skip_wells is not None else ()
    if masks:
        img_type = 'mask'
    if ref_wells is not None:
        ref_wells = list(ref_wells)

    if conditions2remove is not None:
        conditions2remove = [tuple(condition.split(',')) for condition in conditions2remove]

    if field_idx is not None:
        for idx in field_idx:
            assert idx >= 0, 'field_idx must be a non-negative integer'

    # Set color palette for treatments
    plate_layout = pd.read_csv(plate_layout, sep=",|;", engine='python')
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
            normalize_mode=normalize_mode,
            pmin=percentiles[0],
            pmax=percentiles[1],
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
@click.option('--filename_well_idx', type=str, default='4 7',
              callback=split_ws(item_type=int, expected_count=2),
              help='Well-name slice start/stop indices in filename, e.g. --filename_well_idx "4 7"')
@click.option('--filename_field_idx', type=str, default='17 21',
              callback=split_ws(item_type=int, expected_count=2),
              help='Field slice start/stop indices in filename, e.g. --filename_field_idx "17 21"')
@click.option('--skip_wells', type=str, default=None, callback=split_ws(item_type=str),
              help='Wells to skip, e.g. --skip_wells "A01 H12"')
@click.option('--cmap', type=str, default='cet_glasbey',
              help='Color map to use for boxplots')
@click.option('--condition_vars', type=str, default='Treat Dose', callback=split_ws(item_type=str),
              help='Variables for palette conditions, e.g. --condition_vars "Treat Dose"')
@click.option('--conditions2remove', type=str, default=None, callback=split_ws(item_type=str),
              help=('Condition tuples to drop from palette; each token is comma-joined vars, '
                    'e.g. --conditions2remove "DMSO,L OtherTreat,H"'))
@click.option('--check_batches', is_flag=True,
              help='Check if input_path contains subdirectories')
@click.option('--y_label', type=str, default='Object Count',
              help='Label for y-axis')
@click.option('--y_lim', type=str, default=None,
              callback=split_ws(item_type=float, expected_count=2),
              help='Y-axis limits min max, e.g. --y_lim "0 1000"')
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

    skip_wells = skip_wells if skip_wells is not None else ()

    if conditions2remove is not None:
        conditions2remove = [tuple(condition.split(',')) for condition in conditions2remove]

    # Set color palette for treatments
    plate_layout = pd.read_csv(plate_layout, sep=",|;", engine='python')
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
