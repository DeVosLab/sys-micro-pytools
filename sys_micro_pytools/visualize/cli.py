import click
from functools import partial
from .channel_plots import create_channel_plots

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
@click.option('--img_type', type=click.Choice(['multichannel', 'grayscale']), required=True,
              help='Type of image to plot. Options: multichannel, grayscale')
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

if __name__ == '__main__':
    cli()