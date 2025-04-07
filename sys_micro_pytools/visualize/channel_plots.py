from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm
import tifffile
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from typing import Union, List, Tuple, Optional

from sys_micro_pytools.io import read_tiff_or_nd2
from sys_micro_pytools.preprocess.flat_field import flat_field_correction
from sys_micro_pytools.preprocess.normalize import normalize_img, normalize_per_channel, get_ref_wells_percentiles
from sys_micro_pytools.preprocess.composite import create_composite


def create_channel_plots(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    img_type: str = 'multichannel',
    channels2use: Union[List[int], int] = 0,
    suffix: str = '.nd2',
    normalize: bool = False,
    ref_wells: Optional[List[str]] = None,
    filename_well_idx: Tuple[int, int] = (4, 7),
    filename_field_idx: Tuple[int, int] = (17, 21),
    flat_field_path: Optional[Union[str, Path]] = None,
    percentiles: Tuple[float, float] = (0.1, 99.9),
    pattern2ignore: Optional[List[str]] = None,
    patterns2have: Optional[List[str]] = None,
    field_idx: Optional[int] = None,
    output_type: List[str] = ['channels', 'composite']
) -> None:
    """Create channel plots and composite images from multichannel microscopy data.

    Args:
        input_path: Path to directory containing images
        output_path: Path to directory where output will be saved
        img_type: Type of image to plot ('multichannel' or 'grayscale')
        channels2use: Channels to use for making the plots
        suffix: Suffix of image files
        normalize: Whether to store normalized channel images
        ref_wells: Well(s) to use as reference for normalization
        filename_well_idx: Start and stop indices of well name in filename
        filename_field_idx: Start and stop indices of field number in filename
        flat_field_path: Path to flat field correction image
        percentiles: Percentiles for image normalization
        pattern2ignore: Pattern to ignore in filename
        patterns2have: Patterns required in filename
        field_idx: Index of field to plot
        output_type: Type of output to save ('channels' and/or 'composite')
    """
    # Convert paths to Path objects
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    # Create output directories
    output_path.mkdir(parents=True, exist_ok=True)
    if 'channels' in output_type:
        output_path_channels = output_path.joinpath('plots_channels')
        output_path_channels.mkdir(exist_ok=True)
    if 'composite' in output_type:
        output_path_composite = output_path.joinpath('plots_composite')
        output_path_composite.mkdir(exist_ok=True)

    # Load flat field
    if flat_field_path is not None:
        flat_field_path = Path(flat_field_path)
        if flat_field_path.suffix == '.tif':
            flat_field_files = [flat_field_path]
        else:
            flat_field_files = sorted([f for f in flat_field_path.iterdir() if \
                                    f.is_file() and f.suffix == '.tif' and not f.stem.startswith('.')])     
            flat_field_files = [f for f in flat_field_files if 'FF'in f.stem] 
        flat_field_file = [f for f in flat_field_files if Path(input_path).stem in f.stem][0]
        flat_field = tifffile.imread(str(flat_field_file)).astype(float)
    else:
        flat_field = None
    
    # Loop over files and save plots
    files = sorted([f for f in input_path.iterdir() if f.suffix == suffix and not f.name.startswith('.')])
    wells = [f.stem[filename_well_idx[0]:filename_well_idx[1]] for f in files]
    fields = [f.stem[filename_field_idx[0]:filename_field_idx[1]] for f in files]

    df = pd.DataFrame({'file': files, 'well': wells, 'field': fields})

    if field_idx is not None:
        idx = [i for i, f in enumerate(fields) if int(f) == field_idx]
        files = [files[i] for i in idx]
        wells = [wells[i] for i in idx]
        fields = [fields[i] for i in idx]
    
    if ref_wells is not None:
        pmin_vals, pmax_vals = get_ref_wells_percentiles(
            df, ref_wells=ref_wells, 
            n_channels=len(channels2use), 
            flat_field=flat_field, 
            pmin=percentiles[0], pmax=percentiles[1]
            )

    for file, well, field in tqdm(zip(files, wells, fields), total=len(files)):
        if pattern2ignore is not None and any([x in str(file.stem) for x in pattern2ignore]):
            continue

        if patterns2have is not None and not any([x in str(file.stem) for x in patterns2have]):
            continue

        img = read_tiff_or_nd2(file, bundle_axes='cyx' if img_type == 'multichannel' else 'yx').astype(float)
        
        if flat_field is not None:
            img = flat_field_correction(img, flat_field)

        if img_type == 'grayscale':
            n_channels = 1
            img_norm = normalize_img(
                img, 
                pmin=percentiles[0], pmax=percentiles[1],
                pmin_val=pmin_vals[0] if ref_wells is not None else None, 
                pmax_val=pmax_vals[0] if ref_wells is not None else None,
                clip=True
                )
            img_composite = img_norm
        else:
            if channels2use is not None:
                img = img[channels2use,:,:]
            if img.ndim == 2:
                img = np.expand_dims(img, axis=0)
            n_channels = img.shape[0]
            img_norm = img.copy()
            img_norm = normalize_per_channel(
                img,
                pmin=percentiles[0], pmax=percentiles[1],
                pmin_vals=pmin_vals if ref_wells is not None else None, 
                pmax_vals=pmax_vals if ref_wells is not None else None,
                clip=True
                )
            img_composite = create_composite(img_norm, channel_dim=0)
        
        # Save channel plots
        if 'channels' in output_type:
            _, axes = plt.subplots(1, n_channels, figsize=(4*n_channels, 5))
            axes = [axes] if n_channels == 1 else axes # Make sure axes is a list
            for C in range(n_channels):
                # Plot of all channels
                axes[C].imshow(
                    img_norm[C,:,:] if normalize else img[C,:,:],
                    cmap='gray',
                    vmin=0 if not normalize else None,
                    vmax=1 if not normalize else None
                    )
                axes[C].set_xticks([])
                axes[C].set_yticks([])
                axes[C].set_title(f'Channel {C}')


                # Individual channel images
                filename = Path(output_path_channels).joinpath(f'{file.stem}_ch{C:02d}.tif')
                filename.parent.mkdir(exist_ok=True)
                tifffile.imwrite(
                    str(filename),
                    (img_norm[C,:,:] if normalize else img[C,:,:]).astype(np.float32),
                    imagej=True,
                    compression='zlib'
                    )
            plt.suptitle(f'{well} {field}')
            plt.tight_layout()
            filename = Path(output_path_channels).joinpath(f'{file.stem}.jpg')
            plt.savefig(filename, dpi=600)
            plt.close()

        # Save composite plots
        if 'composite' in output_type:
            _, axes = plt.subplots(1, 1, figsize=(9, 9))
            filename = Path(output_path_composite).joinpath(f'{file.stem}.tif')
            filename.parent.mkdir(exist_ok=True)
            img_composite = np.moveaxis(img_composite.astype(np.float32), 0, 2)
            tifffile.imwrite(
                str(filename),
                img_composite,
                photometric='rgb',
                )

def main(**kwargs):
    """Main function that processes command line arguments and calls create_channel_plots."""
    create_channel_plots(
        input_path=kwargs['input_path'],
        output_path=kwargs['output_path'],
        img_type=kwargs['img_type'],
        channels2use=kwargs['channels2use'],
        suffix=kwargs['suffix'],
        normalize=kwargs['normalize'],
        ref_wells=kwargs['ref_wells'],
        filename_well_idx=kwargs['filename_well_idx'],
        filename_field_idx=kwargs['filename_field_idx'],
        flat_field_path=kwargs['flat_field_path'],
        percentiles=kwargs['percentiles'],
        pattern2ignore=kwargs['pattern2ignore'],
        patterns2have=kwargs['patterns2have'],
        field_idx=kwargs['field_idx'],
        output_type=kwargs['output_type']
    )

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('-i', '--input_path', type=str, default=None,
        help='Path to directory containing images')
    parser.add_argument('-o', '--output_path', type=str, default=None,
        help='Path to directory where output will be saved')
    parser.add_argument('--img_type', type=str, default='multichannel',
        choices=['multichannel', 'grayscale'], 
        help='Type of image to plot. Options: multichannel, grayscale')
    parser.add_argument('--channels2use', type=int, nargs='+', default=0,
        help='Channels to use for making the plots')
    parser.add_argument('--suffix', type=str, default='.nd2',
        help='Suffix of image files')
    parser.add_argument('--normalize', action='store_true',
        help='Store normalized channel images. Composite images are always normalized')
    parser.add_argument('--ref_wells', type=str, nargs='+', default=None,
        help='Well(s) to use as reference for normalization using its percentiles')
    parser.add_argument('--filename_well_idx', nargs=2, type=int, default=[4,7],
        help='Start and stop indices of the well name in the filename (default: [4,7])')
    parser.add_argument('--filename_field_idx', nargs=2, type=int, default=[17,21],
        help='Start and stop indices of the field number in the filename (default: [17,21])')
    parser.add_argument('-f', '--flat_field_path', type=str, default=None,
        help='Path to image to use as flat field correction')
    parser.add_argument('--percentiles', nargs=2, type=float, default=[0.1, 99.9],
        help='Percentiles to use for image normalization (default: [0.1, 99.9])')
    parser.add_argument('--pattern2ignore', type=str, default=None,
        help='Pattern to ignore in filename')
    parser.add_argument('--patterns2have', type=str, nargs='+', default=None,
        help='Patterns to have in filename. If not None, only files with these patterns will be processed')
    parser.add_argument('--field_idx', type=int, default=None,
        help='Index of field to plot (default: None)')
    parser.add_argument('--output_type', type=str, nargs='+', default=['channels', 'composite'],
        help='Type of output to save (default: channels and composite)')
    args = parser.parse_args()

    if isinstance(args.channels2use, int):
        args.channels2use = [args.channels2use]

    if args.ref_wells is not None:
        if isinstance(args.ref_wells, str):
            args.ref_wells = [args.ref_wells]
        else:
            args.ref_wells = [str(well) for well in args.ref_wells]

    if isinstance(args.pattern2ignore, str):
        args.pattern2ignore = [args.pattern2ignore]

    if isinstance(args.patterns2have, str):
        args.patterns2have = [args.patterns2have]
    
    if isinstance(args.output_type, str):
        args.output_type = [args.output_type]

    return args


if __name__ == '__main__':
    args = parse_args()
    main(**vars(args))

