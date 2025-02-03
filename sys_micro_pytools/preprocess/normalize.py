from typing import Union, List, Tuple
import numpy as np
import pandas as pd
from numbers import Number

from io import read_tiff_or_nd2
from preprocess.flat_field import flat_field_correction

def normalize_img(img: np.ndarray, pmin: float=0.1, pmax: float=99.9,
                  pmin_val: float=None, pmax_val: float=None,
                  eps: float=1e-9, clip: bool=True) -> np.ndarray:
    '''Normalize an image to a range of [0, 1]
    
    Args:
        img (np.ndarray): The image to normalize.
        pmin (float): The minimum percentile to use for normalization. Gets overridden by pmin_val if provided.
        pmax (float): The maximum percentile to use for normalization. Gets overridden by pmax_val if provided.
        pmin_val (float): The minimum value to use for normalization. If None, use the pmin percentile.
        pmax_val (float): The maximum value to use for normalization. If None, use the pmax percentile.
        eps (float): A small value to add to the denominator to avoid division by zero.
        clip (bool): Whether to clip the normalized image to the range [0, 1].
    
    Returns:
        np.ndarray: The normalized image.
    '''
    if pmin_val is None:
        pmin_val = np.percentile(img, pmin)
    else:
        if isinstance(pmin_val, np.ndarray):
            pmin_val = pmin_val[0]
    if pmax_val is None:
        pmax_val = np.percentile(img, pmax)
    else:
        if isinstance(pmax_val, np.ndarray):
            pmax_val = pmax_val[0]
    img = (img - pmin_val)/ (pmax_val - pmin_val + eps)
    if clip:
        img = np.clip(img, 0, 1)
    return img

def normalize_per_channel(img: np.ndarray,
                          pmin: float=0.1, 
                          pmax: float=99.9,
                          pmin_vals: Union[float, List, Tuple]=None,
                          pmax_vals: Union[float, List, Tuple]=None,
                          eps: float=1e-9, clip: bool=True) -> np.ndarray:
    '''Normalize an image per channel
    
    Args:
        img (np.ndarray): The image to normalize.
        pmin (float): The minimum percentile to use for normalization. Gets overridden by pmin_val if provided.
        pmax (float): The maximum percentile to use for normalization. Gets overridden by pmax_val if provided.
        pmin_vals (float, List, Tuple): The minimum value to use for normalization. If None, use the pmin percentile.
        pmax_vals (float, List, Tuple): The maximum value to use for normalization. If None, use the pmax percentile.
        eps (float): A small value to add to the denominator to avoid division by zero.
        clip (bool): Whether to clip the normalized image to the range [0, 1].
    
    Returns:
        np.ndarray: The normalized image.
    '''
    if not isinstance(img, np.ndarray):
        raise TypeError('img must be a numpy array')
    if len(img.shape) != 3:
        raise ValueError('img must be a 3D array with shape (C, H, W)')
    
    n_channels = img.shape[0]
    if pmin_vals is not None:
        if isinstance(pmin_vals, Number):
            pmin_vals = [pmin_vals]*n_channels
        elif len(pmin_vals) != n_channels:
            raise ValueError(f'pmin_vals must be a float or a list/tuple of length {n_channels}')
    else:
        pmin_vals = [None]*n_channels
    
    if pmax_vals is not None:
        if isinstance(pmax_vals, Number):
            pmax_vals = [pmax_vals]*n_channels
        elif len(pmax_vals) != n_channels:
            raise ValueError(f'pmax_vals must be a float or a list/tuple of length {n_channels}')
    else:
        pmax_vals = [None]*n_channels

    for C, pmin_val, pmax_val in zip(range(n_channels), pmin_vals, pmax_vals):
        img[C,:,:] = normalize_img(
            img[C,:,:], 
            pmin=pmin, pmax=pmax,
            pmin_val=pmin_val, pmax_val=pmax_val,
            eps=eps,
            clip=clip
        )
    return img

def get_ref_wells_percentiles(df: pd.DataFrame, ref_wells: List[str], 
                                 n_channels: int, flat_field: np.ndarray, 
                                 pmin: float=0.1, pmax: float=99.9,
                                 path_col: str='file', well_col: str='well'):
    if path_col not in df.columns:
        raise ValueError(f'{path_col} not in df columns')
    if well_col not in df.columns:
        raise ValueError(f'{well_col} not in df columns')
    
    df_ref = df[df[well_col].isin(ref_wells)]
    pmin_vals = np.zeros((n_channels,))
    pmax_vals = np.zeros((n_channels,))
    
    count = 0
    for _, row in df_ref.iterrows():
        file = str(row[path_col])
        ref_img = read_tiff_or_nd2(file, bundle_axes='yx' if n_channels == 1 else 'cyx').astype(float)
        if flat_field is not None:
            ref_img = flat_field_correction(ref_img, flat_field)
        if ref_img.ndim == 2:
            ref_img = np.expand_dims(ref_img, axis=0)
        for C in range(n_channels):
            pmin_vals[C] += np.percentile(ref_img[C,:,:], pmin)
            pmax_vals[C] += np.percentile(ref_img[C,:,:], pmax)
        count += 1
    pmin_vals /= count
    pmax_vals /= count
    return pmin_vals, pmax_vals