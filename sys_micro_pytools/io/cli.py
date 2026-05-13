import click

from sys_micro_pytools.cli_utils import split_ws

from .io import split_lif, split_lif_bioformats

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


@io.command(name='split-lif-bf')
@click.argument('input_path', type=click.Path(exists=True, dir_okay=False))
@click.option('-o', '--output_dir', type=click.Path(), required=True,
              help='Path to the output directory')
@click.option('--compression', type=click.STRING, default='LZW',
              help='Bio-Formats compression (e.g., LZW, ZLIB). Default is LZW.')
@click.option('--series', type=str, default=None, callback=split_ws(item_type=int),
              help=('Series indices to export as whitespace-separated ints, e.g. --series "0 1 2". '
                    'If omitted, auto-detect via showinf.'))
@click.option('--bfconvert_bin', type=click.STRING, default='bfconvert',
              help='Path or command name for bfconvert executable.')
@click.option('--showinf_bin', type=click.STRING, default='showinf',
              help='Path or command name for showinf executable.')
@click.option('--subdir/--no-subdir', default=True,
              help='Whether to create a subdirectory in output_dir with the name of the lif file.')
def split_lif_bf_cli(input_path, output_dir, compression, series, bfconvert_bin, showinf_bin, subdir):
    """Split a .lif file by series using Bio-Formats bfconvert."""
    series_list = list(series) if series else None
    split_lif_bioformats(
        input_path,
        output_dir,
        compression=compression,
        do_create_subdir=subdir,
        series=series_list,
        bfconvert_bin=bfconvert_bin,
        showinf_bin=showinf_bin,
    )

if __name__ == '__main__':
    io()
