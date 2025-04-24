from typing import Union, Literal
from pathlib import Path
import numpy as np
import pandas as pd
import re
import random
from skimage.transform import resize
from skimage.color import label2rgb
import colorcet as cc
from matplotlib import pyplot as plt

from sys_micro_pytools.io import read_tiff_or_nd2
from sys_micro_pytools.preprocess.normalize import normalize_img, normalize_per_channel, get_ref_wells_percentiles
from sys_micro_pytools.preprocess.flat_field import flat_field_correction
from sys_micro_pytools.preprocess.composite import create_composite


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
        condition_vars: Union[list, tuple, str],
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

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2 * n_cols + 3, 2 * n_rows))
    count = 0
    conditions = []
    for r, row in enumerate(rows):
        for c, col in enumerate(cols):
            df_row_col = df[(df['Row'] == row) & (df['Col'] == col)].sort_values(by='Field')
            if df_row_col.shape[0] == 0:
                continue
            condition_vals = tuple(df_row_col[hue_var].values[0] for hue_var in condition_vars)
            condition_color = palette[condition_vals]
            conditions.append(condition_vals)

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
                    pmin_vals=pmin_vals, pmax_vals=pmax_vals,
                    clip=True,
                )
                img = create_composite(img, channel_dim=0)
                img = np.moveaxis(img, 0, -1)
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
            axes[r, c].patch.set_edgecolor(condition_color)  
            axes[r, c].patch.set_linewidth(10)

    # Add a legend on the right side
    legend_elements = []
    for condition, color in palette.items():
        if condition not in conditions:
            continue

        # Format the condition label
        if isinstance(condition, tuple):
            label = ', '.join(str(c) for c in condition)
        else:
            label = str(condition)
        legend_elements.append(plt.Rectangle((0, 0), 1, 1, color=color, label=label))
    
    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0, 0.90, 1])  # Adjust layout to leave space for legend
    fig.legend(handles=legend_elements, 
               loc='center right', 
               title=' - '.join(condition_vars),
               fontsize=14,
               bbox_to_anchor=(.95, 0.5),
               )
    return fig