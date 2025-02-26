from argparse import ArgumentParser
from typing import Union, Tuple, List, Literal
from pathlib import Path
from tqdm import tqdm
import itertools
import numpy as np
import pandas as pd
import re
import random
import tifffile
from skimage.transform import resize
from skimage.color import label2rgb
import colorcet as cc
from matplotlib import pyplot as plt
from matplotlib import colormaps

from io import read_tiff_or_nd2
from preprocess.normalize import normalize_img, normalize_per_channel, get_ref_wells_percentiles
from preprocess.flat_field import get_flat_field_files, flat_field_correction
from preprocess.composite import create_composite2D
from df import link_df2plate_layout
from visualize import create_palette


def get_df_images(input_path: Union[str, Path], check_batches: bool, suffix: str,
                  filename_well_idx: Union[list, tuple], filename_field_idx: Union[list, tuple],
                  skip_wells: Union[list, tuple]) -> pd.DataFrame:
    ''' Get dataframe with image information

    Parameters
    ----------
    input_path : str or Path
        Path to directory containing images
    check_batches : bool
        Check if input_path contains subdirectories
    suffix : str
        Suffix of image files
    filename_well_idx : list or tuple
        Start and stop indices of the well name in the filename
    filename_field_idx : list or tuple
        Start and stop indices of the field number in the filename
    skip_wells : list or tuple
        List of wells to skip

    Returns
    -------
    df_images : pd.DataFrame
        Dataframe with image information

    '''
    assert isinstance(input_path, (str, Path)), 'input_path must be a string or Path object'
    assert isinstance(check_batches, bool), 'check_batches must be a boolean'
    assert isinstance(suffix, str), 'suffix must be a string'
    assert isinstance(filename_well_idx, (list, tuple)), 'filename_well_idx must be a list or tuple'
    assert isinstance(filename_field_idx, (list, tuple)), 'filename_field_idx must be a list or tuple'
    assert isinstance(skip_wells, (list, tuple)), 'skip_wells must be a list or tuple'

    input_path = Path(input_path)
    dirs = [d for d in input_path.iterdir() if d.is_dir()] if check_batches else [input_path]
    if dirs == []:
        dirs = [input_path]
    df_images = pd.DataFrame()
    for dir in dirs:
        # Assume DATE_CELLTYPE_PLATEx_Rx format
        cell_type = dir.stem.split('_')[1]

        # Use re.findall to extract numbers after "plate" and "R"
        plate_pattern = r'plate([A-Za-z0-9]+)'
        r_pattern = r'R(\d+)'
        plate_id = re.findall(plate_pattern, dir.stem)[0] if 'plate' in dir.stem else 1
        rep_id = re.findall(r_pattern, dir.stem)[0] if dir.stem[-2] == 'R' else 1

        files = sorted([f for f in dir.iterdir() if f.suffix == suffix and not f.name.startswith('.')])
        wells = [f.stem[filename_well_idx[0]:filename_well_idx[1]] for f in files]
        rows =  [w[0] for w in wells]
        cols = [int(w[1:]) for w in wells]
        fields = [f.stem[filename_field_idx[0]:filename_field_idx[1]] for f in files]

        df_entry = pd.DataFrame().from_dict({
            'Dir': [dir.stem] * len(files),
            'CellType': [cell_type] * len(files),
            'Rep': [int(rep_id)] * len(files),
            'Plate': [plate_id] * len(files),
            'Well': wells,
            'Row': rows,
            'Col': cols,
            'Field': fields,
            'Filename': files
        })

        df_images = pd.concat(
            [df_images if not df_images.empty else None, df_entry],
            axis=0,
            ignore_index=True
            ).reset_index(drop=True)

    # Remove wells to skip
    df_images = df_images[~df_images['Well'].isin(skip_wells)].reset_index(drop=True)

    return df_images

def create_grid_plot(
        df: pd.DataFrame,
        flat_field: np.ndarray,
        hue_vars: Union[list, tuple, str],
        palette: dict,
        field_idx: Union[list, tuple, int]=None,
        img_type: Literal['multi_channel', 'grayscale', 'mask']='multi_channel',
        channels2use: Union[list, tuple, int]=None,
        ref_wells: Union[list, tuple, str]=None,
        pmin: float=0.1, pmax: float=99.9,
        title: str=None,
    ):
    rows = sorted(df['Row'].unique())
    cols = sorted(df['Col'].unique())
    n_rows = len(rows)
    n_cols = len(cols)
    if field_idx is not None:
        if isinstance(field_idx, int):
            field_idx = [field_idx]
        if len(field_idx) == 1:
            field_idx = n_rows * n_cols * field_idx
        else:
            assert len(field_idx) == n_rows * n_cols, \
                'field_idx must be a single integer or a list/tuple of length n_rows * n_cols'
            field_idx = field_idx

    
    if ref_wells is not None:
        pmin_vals, pmax_vals = get_ref_wells_percentiles(
            df, ref_wells=ref_wells, 
            n_channels=len(channels2use), 
            flat_field=flat_field, 
            pmin=pmin, pmax=pmax,
            path_col='Filename', well_col='Well'
        )
    else:
        pmin_vals = None
        pmax_vals = None

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2 * n_cols, 2 * n_rows))
    count = 0
    for r, row in enumerate(rows):
        for c, col in enumerate(cols):
            df_row_col = df[(df['Row'] == row) & (df['Col'] == col)].sort_values(by='Field')
            if df_row_col.shape[0] == 0:
                continue
            hue_var_vals = tuple(df_row_col[hue_var].values[0] for hue_var in hue_vars)
            hue = palette[hue_var_vals]

            # Randomly select a field from the well
            if field_idx is not None:
                idx = field_idx[count]
                count += 1
            else:
                idx = random.randint(0, df_row_col.shape[0]-1)
            
            # Check if idx is within the range of the number of files
            if idx >= df_row_col.shape[0]:
                img = flat_field
            else:
                # Load image
                filename = df_row_col['Filename'].values[idx]
                img = read_tiff_or_nd2(str(filename), bundle_axes='cyx' if \
                                      img_type == 'multi_channel' else 'yx').astype(float)
            
            if flat_field is not None and img_type != 'mask':
                img = flat_field_correction(img, flat_field)

            # Resize image
            shape = (224, 224)
            order = 1
            if img_type == 'mask':
                order = 0
            elif img_type == 'multi_channel':
                n_channels = len(channels2use)
                img = img[channels2use,:,:]
                shape = (n_channels,) + shape
            img = resize(img, shape, order=order, anti_aliasing=True)

            # Normalize image
            if img_type == 'grayscale':
                img = normalize_img(
                    img, pmin=pmin, pmax=pmax,
                    pmin_val=pmin_vals, pmax_val=pmax_vals,
                    clip=True,
                )
            elif img_type == 'multi_channel':
                if channels2use is not None:
                    img = img[channels2use,:,:]
                if img.ndim == 2:
                    img = np.expand_dims(img, axis=0)
                img = normalize_per_channel(
                    img, pmin=pmin, pmax=pmax,
                    pmin_val=pmin_vals, pmax_val=pmax_vals,
                    clip=True,
                )
                img = create_composite2D(img, channel_dim=0)
            elif img_type == 'mask':
                pass
            else:
                raise ValueError(f'img_type {img_type} is not supported')
            
            # Label to rgb
            if img_type == 'mask':
                img = label2rgb(img, bg_label=0, bg_color=(0, 0, 0))

            axes[r, c].imshow(img)
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
            axes[r, c].patch.set_edgecolor(hue)  
            axes[r, c].patch.set_linewidth(10)
    plt.suptitle(title)
    plt.tight_layout()
    return fig

def main(**kwargs):
    input_path = Path(kwargs['input_path'])
    output_path = Path(kwargs['output_path'])
    output_path.mkdir(parents=True, exist_ok=True)

    if isinstance(kwargs['skip_wells'], str):
        kwargs['skip_wells'] = [kwargs['skip_wells']]

    if isinstance(kwargs['hue_vars'], str):
        kwargs['hue_vars'] = [kwargs['hue_vars']]

    if kwargs['grayscale'] or kwargs['masks']:
        assert kwargs['grayscale'] != kwargs['masks'], \
            'Either --grayscale or --masks must be specified, but not both'
    
    if kwargs['remove_hue_combo'] is not None:
        kwargs['remove_hue_combo'] = [tuple(hue_combo.split(',')) for hue_combo in kwargs['remove_hue_combo']]

    if isinstance(kwargs['field_idx'], int):
        kwargs['field_idx'] = [kwargs['field_idx']]
    for idx in kwargs['field_idx']:
        assert idx >= 0, 'field_idx must be a positive integer'

    # Set color palette for treatments
    plate_layout = pd.read_csv(plate_layout, sep=",|;"); 
    plate_layout.columns = [col.capitalize() for col in plate_layout.columns]
    plate_layout['Plate'] = plate_layout['Plate'].astype(str)

    # Set color palette
    palette, _ = create_palette(
        plate_layout,
        kwargs['hue_vars'],
        cmap=kwargs['cmap'],
        remove_hue_combo=kwargs['remove_hue_combo'],
        hue_start_rgba=kwargs['hue_start_rgba']
    )

    # Get flat field images
    if kwargs['flat_field_path'] is not None:
        flat_field_files = get_flat_field_files(kwargs['flat_field_path'])
    else:
        flat_field_files = None
    
    # Image path
    input_path = Path(kwargs['input_path'])
    df_images = get_df_images(
        input_path,
        kwargs['check_batches'],
        kwargs['suffix'],
        kwargs['filename_well_idx'],
        kwargs['filename_field_idx'],
        kwargs['skip_wells']
    )
    df_images.columns = [col.capitalize() for col in df_images.columns]
    df_images['Plate'] = df_images['Plate'].astype(str)

    # Merge plate layout with df_images
    df_images = link_df2plate_layout(df_images, plate_layout)
    print(df_images.head())

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
            palette,
            field_idx=kwargs['field_idx'],
            img_type=kwargs['img_type'],
            channels2use=kwargs['channels2use'],
            title=dir
        )

        # Save figure
        filename = Path(output_path).joinpath(f'{dir}.jpg')
        filename.parent.mkdir(exist_ok=True, parents=True)
        print(f'Saving figure to {filename}')
        fig.savefig(filename, dpi=600)
        plt.close()


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('-i', '--input_path', type=str, default=None,
        help='Path to directory containing images')
    parser.add_argument('-o', '--output_path', type=str, default=None,
        help='Path to directory where output will be saved')
    parser.add_argument('--plate_layout', type=str, default=None,
        help='Path to plate layout file')
    parser.add_argument('--suffix', type=str, default='.nd2',
        help='Suffix of image files')
    parser.add_argument('--filename_well_idx', nargs=2, type=int, default=[4,7],
        help='Start and stop indices of the well name in the filename (default: [4,7])')
    parser.add_argument('--filename_field_idx', nargs=2, type=int, default=[17,21],
        help='Start and stop indices of the field number in the filename (default: [17,21])')
    parser.add_argument('--skip_wells', nargs='+', type=str, default=[],
        help='List of wells to skip')
    parser.add_argument('--img_type', type=str, default='multi_channel',
        choices=['multi_channel', 'grayscale', 'mask'],
        help='Type of image to plot')
    parser.add_argument('--channels2use', type=int, nargs='+', default=None,
        help='Channels to use for making the plots')
    parser.add_argument('--ref_wells', type=str, nargs='+', default=None,
        help='Well(s) to use as reference for normalization using its percentiles')
    parser.add_argument('--masks', action='store_true',
        help='Indicate that image is a mask. FF correction will not be applied')
    parser.add_argument('--flat_field_path', type=str, default=None,
        help='Path to image to use as flat field correction')
    parser.add_argument('--cmap', type=str, default='tab10',
        help='Color map to use image borders in grid plot')
    parser.add_argument('--hue_vars', nargs='*', type=str, default=['Treat', 'Dose'],
        help='Variables to use for color encoding')
    parser.add_argument('--remove_hue_combo', nargs='*', type=str, default=None,
        help='List of hue combinations to remove from the hue palette, e.g. --remove_hue_combo "DMSO,L"')
    parser.add_argument('--hue_start_rgba', nargs=4, type=float, default=None,
        help='RGBA values for the first hue in the palette')
    parser.add_argument('--check_batches', action='store_true',
        help='Check if input_path contains subdirectories')
    parser.add_argument('--field_idx', type=int, nargs='+', default=None,
        help='Index of the field to plot for each well. If None, a random field will be selected')
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args() 
    main(**vars(args))