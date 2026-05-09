import click
from pathlib import Path
import tifffile
from sys_micro_pytools.preprocess.flat_field import compute_flat_field
from sys_micro_pytools.preprocess.split_nd2 import split_nd2, split_nd2_folder


def empty_to_none(ctx, param, value):
    if value == ():
        return None
    return value

@click.group()
def preprocess():
    pass

@preprocess.command(name='flat-field')
@click.option('-i', '--input_path', type=click.Path(exists=True), required=True, 
              help='Path to the input images')
@click.option('-o', '--output_path', type=click.Path(), required=True,
              help='Path to save the flat field')
@click.option('--suffix', type=click.STRING, default='.nd2',
              help='Suffix of the input images')
@click.option('--sigma', type=click.FLOAT, default=90,
              help='Sigma of the Gaussian blur filter')
@click.option('--n_images', type=click.INT, default=None,
              help='Number of images to use to generate the flat field')
@click.option('--from_well', type=click.STRING, default=None,
              help='Well to use to generate the flat field. If None, use n random images from the input path to generate the flat field')
@click.option('--skip_wells', type=click.STRING, multiple=True, callback=empty_to_none,
              help='Wells to ignore when generating the flat field. If None, use all wells.')
@click.option('--grayscale', is_flag=True, default=False,
              help='Use grayscale images instead of RGB images.')
@click.option('--method', type=click.Choice(['mean', 'median']), default='mean',
              help='Method to use to generate the flat field.')
@click.option('--seed', type=click.INT, default=0,
              help='Seed for random number generator')
def flat_field(input_path, output_path, suffix, sigma, n_images, from_well, 
               skip_wells, grayscale, method, seed):
    # Compute flat field
    input_path = Path(input_path)
    flat_field, type = compute_flat_field(
        input_path,
        from_well=from_well,
        skip_wells=skip_wells,
        suffix=suffix,
        n_images=n_images,
        sigma=sigma,
        grayscale=grayscale,
        method=method,
        seed=seed
    )

    # Save flat field
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    filename = f'FF_{input_path.stem}_{type}.tif'
    tifffile.imwrite(
        output_path.joinpath(filename),
        flat_field,
        compression='zlib',
        imagej=True,
        metadata={'axes': 'CYX' if not grayscale else 'YX'}
    )

@preprocess.command(name='split-nd2')
@click.option('-i', '--input_path', type=click.Path(exists=True), required=True,
              help='Path to a .nd2 file or a folder containing .nd2 files.')
@click.option('-o', '--output_path', type=click.Path(), required=True,
              help='Folder where the per-FOV .tif files will be written.')
@click.option('--bundle_axes', type=click.STRING, default='cyx', show_default=True,
              help="Axes to bundle per FOV (e.g. 'cyx', 'yx', 'czyx'). The 'v' "
                   "axis is always iterated over and is ignored if passed.")
@click.option('--compression', type=click.STRING, default='zlib', show_default=True,
              help='Compression for the output .tif files (e.g. zlib, lzw).')
@click.option('--overwrite/--no-overwrite', default=False, show_default=True,
              help='Whether to overwrite existing .tif files in the output folder.')
def split_nd2_cli(input_path, output_path, bundle_axes, compression, overwrite):
    """Split multi-FOV .nd2 file(s) into one .tif per field of view (point).

    Accepts either a single .nd2 file or a folder of .nd2 files. The output
    filenames always contain a 'Point<WellCode>_<FOVIndex>' token following
    the Nikon NIS-Elements naming convention.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    if input_path.is_file():
        split_nd2(
            input_path,
            output_path,
            bundle_axes=bundle_axes,
            compression=compression,
            overwrite=overwrite,
        )
    else:
        split_nd2_folder(
            input_path,
            output_path,
            bundle_axes=bundle_axes,
            compression=compression,
            overwrite=overwrite,
        )


if __name__ == '__main__':
    preprocess()