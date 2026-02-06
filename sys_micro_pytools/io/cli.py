import click
from .io import split_lif

@click.group()
def io():
    """Input/Output operations."""
    pass

@io.command(name='split-lif')
@click.argument('input_path', type=click.Path(exists=True, dir_okay=False))
@click.option('-o', '--output_dir', type=click.Path(), required=True,
              help='Path to the output directory')
@click.option('--bundle_axes', type=click.STRING, default='TZCYX',
              help='Bundle axes of the images. Default is TZCYX.')
@click.option('--compression', type=click.STRING, default='zlib',
              help='Compression algorithm to use (e.g., zlib, lzw). Default is zlib.')
@click.option('--subdir/--no-subdir', default=True,
              help='Whether to create a subdirectory in output_dir with the name of the lif file.')
def split_lif_cli(input_path, output_dir, bundle_axes, compression, subdir):
    """Split a .lif file with multiple images into individual images stored as compressed .tif files."""
    split_lif(input_path, output_dir, bundle_axes=bundle_axes, compression=compression, do_create_subdir=subdir)

if __name__ == '__main__':
    io()
