from argparse import ArgumentParser
from pathlib import Path
from typing import List
from tqdm import tqdm
import tifffile
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

from io import read_tif_or_nd2
from preprocess.flat_field import flat_field_correction
from preprocess.normalize import normalize_img, normalize_per_channel, get_ref_wells_percentiles

from visualize.composite import create_composite2D


def main(**kwargs):
    # Get input arguments
    input_path = Path(kwargs['input_path'])
    suffix = kwargs['suffix']
    pmin, pmax = kwargs['percentiles']

    # Output path
    output_path = Path(kwargs['output_path'])
    output_path.mkdir(parents=True, exist_ok=True)
    if 'channels' in kwargs['output_type']:
        output_path_channels = output_path.joinpath('plots_channels')
        output_path_channels.mkdir(exist_ok=True)
    if 'composite' in kwargs['output_type']:
        output_path_composite = output_path.joinpath('plots_composite')
        output_path_composite.mkdir(exist_ok=True)

    # Load flat field
    if kwargs['flat_field_path'] is not None:
        flat_field_path = Path(kwargs['flat_field_path'])
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
    wells = [f.stem[kwargs['filename_well_idx'][0]:kwargs['filename_well_idx'][1]] for f in files]
    fields = [f.stem[kwargs['filename_field_idx'][0]:kwargs['filename_field_idx'][1]] for f in files]

    df = pd.DataFrame({'file': files, 'well': wells, 'field': fields})

    if kwargs['field_idx'] is not None:
        idx = [i for i, f in enumerate(fields) if int(f) == kwargs['field_idx']]
        files = [files[i] for i in idx]
        wells = [wells[i] for i in idx]
        fields = [fields[i] for i in idx]
    
    if kwargs['ref_wells'] is not None:
        pmin_vals, pmax_vals = get_ref_wells_percentiles(
            df, ref_wells=kwargs['ref_wells'], 
            n_channels=len(kwargs['channels2use']), 
            flat_field=flat_field, 
            pmin=pmin, pmax=pmax
            )

    for file, well, field in tqdm(zip(files, wells, fields), total=len(files)):
        if kwargs['pattern2ignore'] is not None and any([x in str(file.stem) for x in kwargs['pattern2ignore']]):
            continue

        if kwargs['patterns2have'] is not None and not any([x in str(file.stem) for x in kwargs['patterns2have']]):
            continue

        img = read_tif_or_nd2(file, bundle_axes='cyx' if not kwargs['grayscale'] else 'yx').astype(float)
        
        if flat_field is not None:
            img = flat_field_correction(img, flat_field)

        if kwargs['img_type'] == 'grayscale':
            n_channels = 1
            img_norm = normalize_img(
                img, 
                pmin=pmin, pmax=pmax,
                pmin_val=pmin_vals[0] if kwargs['ref_wells'] is not None else None, 
                pmax_val=pmax_vals[0] if kwargs['ref_wells'] is not None else None,
                clip=True
                )
            img_composite = img_norm
        else:
            if kwargs['channels2use'] is not None:
                img = img[kwargs['channels2use'],:,:]
            if img.ndim == 2:
                img = np.expand_dims(img, axis=0)
            n_channels = img.shape[0]
            img_norm = img.copy()
            img_norm = normalize_per_channel(
                img,
                pmin=pmin, pmax=pmax,
                pmin_vals=pmin_vals if kwargs['ref_wells'] is not None else None, 
                pmax_vals=pmax_vals if kwargs['ref_wells'] is not None else None,
                clip=True
                )
            img_composite = create_composite2D(img_norm, channel_dim=0)
        
        # Save channel plots
        if 'channels' in kwargs['output_type']:
            _, axes = plt.subplots(1, n_channels, figsize=(4*n_channels, 5))
            axes = [axes] if n_channels == 1 else axes # Make sure axes is a list
            for C in range(n_channels):
                # Plot of all channels
                axes[C].imshow(
                    img_norm[C,:,:] if kwargs['normalize'] else img[C,:,:],
                    cmap='gray',
                    vmin=0 if not kwargs['normalize'] else None,
                    vmax=1 if not kwargs['normalize'] else None
                    )
                axes[C].set_xticks([])
                axes[C].set_yticks([])
                axes[C].set_title(f'Channel {C}')


                # Individual channel images
                filename = Path(output_path_channels).joinpath(f'{file.stem}_ch{C:02d}.tif')
                filename.parent.mkdir(exist_ok=True)
                tifffile.imwrite(
                    str(filename),
                    (img_norm[C,:,:] if kwargs['normalize'] else img[C,:,:]).astype(np.float32),
                    imagej=True,
                    compression='zlib'
                    )
            plt.suptitle(f'{well} {field}')
            plt.tight_layout()
            filename = Path(output_path_channels).joinpath(f'{file.stem}.jpg')
            plt.savefig(filename, dpi=600)
            plt.close()

        # Save composite plots
        if 'composite' in kwargs['output_type']:
            _, axes = plt.subplots(1, 1, figsize=(9, 9))
            filename = Path(output_path_composite).joinpath(f'{file.stem}.tif')
            filename.parent.mkdir(exist_ok=True)
            img_composite = np.moveaxis(img_composite.astype(np.float32), 2, 0)
            tifffile.imwrite(
                str(filename),
                img_composite,
                photometric='rgb',
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

