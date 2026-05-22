from typing import Union, List, Tuple, Sequence, Optional, Literal
import warnings
import numpy as np
import pandas as pd
from numbers import Number

from sys_micro_pytools.io import read_tiff_or_nd2
from sys_micro_pytools.preprocess.flat_field import flat_field_correction

NormalizeMode = Literal['off', 'per_image', 'ref_wells', 'global']
NORMALIZE_MODES: Tuple[str, ...] = ('off', 'per_image', 'ref_wells', 'global')


def coerce_channels(channels: Union[int, Sequence[int], None]) -> List[int]:
    if channels is None:
        return [0]
    if isinstance(channels, int):
        return [channels]
    return list(channels)


def select_channel_plane(img: np.ndarray, channel: int) -> np.ndarray:
    '''Extract one channel as a 2D plane from yx or cyx data.'''
    if img.ndim == 2:
        if channel != 0:
            raise ValueError(
                f'Image has no channel axis (shape {img.shape}); '
                f'cannot select channel {channel}'
            )
        return img
    if img.ndim == 3:
        if channel < 0 or channel >= img.shape[0]:
            raise IndexError(
                f'Channel index {channel} out of range for shape {img.shape}'
            )
        return img[channel, :, :]
    raise ValueError(f'Expected 2D or 3D (cyx) image, got shape {img.shape}')

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
                          pmin: Number=0.1, 
                          pmax: Number=99.9,
                          pmin_vals: Union[Number, List, Tuple]=None,
                          pmax_vals: Union[Number, List, Tuple]=None,
                          eps: float=1e-9,
                          clip: bool=True,
                          channel_dim: int=0) -> np.ndarray:
    '''Normalize an image per channel
    
    Args:
        img (np.ndarray): The image to normalize.
        pmin (float): The minimum percentile to use for normalization. Gets overridden by pmin_val if provided.
        pmax (float): The maximum percentile to use for normalization. Gets overridden by pmax_val if provided.
        pmin_vals (float, List, Tuple): The minimum value to use for normalization. If None, use the pmin percentile.
        pmax_vals (float, List, Tuple): The maximum value to use for normalization. If None, use the pmax percentile.
        eps (float): A small value to add to the denominator to avoid division by zero.
        clip (bool): Whether to clip the normalized image to the range [0, 1].
        channel_dim (int): The dimension of the channel.

    Returns:
        np.ndarray: The normalized image.
    '''
    if not isinstance(img, np.ndarray):
        raise TypeError('img must be a numpy array')
    
    n_channels = img.shape[channel_dim]
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
        idx = tuple(slice(None) if i != channel_dim else C for i in range(img.ndim))
        img[idx] = normalize_img(
            img[idx], 
            pmin=pmin, pmax=pmax,
            pmin_val=pmin_val, pmax_val=pmax_val,
            eps=eps,
            clip=clip
        )
    return img


def standardize_img(img: np.ndarray,
                    mean_val: Number=None,
                    std_val: Number=None,
                    eps: float=1e-9) -> np.ndarray:
    '''Standardize an image
    '''
    if mean_val is None:
        mean_val = img.mean()
    if std_val is None:
        std_val = img.std()
    return (img - mean_val) / (std_val + eps)


def standardize_per_channel(img: np.ndarray,
                            mean_vals: Union[Number, List, Tuple]=None,
                            std_vals: Union[Number, List, Tuple]=None,
                            eps: float=1e-9,
                            channel_dim: int=0) -> np.ndarray:
    '''Standardize an image per channel
    '''
    if not isinstance(img, np.ndarray):
        raise TypeError('img must be a numpy array')
    
    n_channels = img.shape[channel_dim]
    if mean_vals is not None:
        if isinstance(mean_vals, Number):
            mean_vals = [mean_vals]*n_channels
        elif len(mean_vals) != n_channels:
            raise ValueError(f'mean_val must be a float or a list/tuple of length {n_channels}')
    else:
        mean_vals = [None]*n_channels
    
    if std_vals is not None:
        if isinstance(std_vals, Number):
            std_vals = [std_vals]*n_channels
        elif len(std_vals) != n_channels:
            raise ValueError(f'std_val must be a float or a list/tuple of length {n_channels}')
    else:
        std_vals = [None]*n_channels

    for C, mean_val, std_val in zip(range(n_channels), mean_vals, std_vals):
        idx = tuple(slice(None) if i != channel_dim else C for i in range(img.ndim))
        img[idx] = standardize_img(img[idx], mean_val=mean_val, std_val=std_val, eps=eps)
    return img



def _ref_wells_provided(ref_wells: Optional[Sequence[str]]) -> bool:
    if not ref_wells:
        return False
    if isinstance(ref_wells, str):
        return bool(ref_wells.strip())
    return len(ref_wells) > 0


def resolve_normalize_mode(
        normalize_mode: str,
        img_type: str,
        ref_wells: Optional[Sequence[str]] = None,
    ) -> NormalizeMode:
    '''Validate ``normalize_mode`` and apply img-type defaults.'''
    if normalize_mode not in NORMALIZE_MODES:
        raise ValueError(
            f'normalize_mode must be one of {NORMALIZE_MODES}, got {normalize_mode!r}'
        )
    if normalize_mode == 'ref_wells' and not _ref_wells_provided(ref_wells):
        raise ValueError('ref_wells is required when normalize_mode is ref_wells')
    if _ref_wells_provided(ref_wells) and normalize_mode != 'ref_wells':
        if img_type == 'multichannel':
            warnings.warn(
                'ref_wells is ignored for multichannel plots; '
                'global normalization is used instead',
                stacklevel=2,
            )
        elif img_type == 'mask':
            warnings.warn(
                'ref_wells is ignored for mask plots',
                stacklevel=2,
            )
        else:
            warnings.warn(
                f'ref_wells is ignored when normalize_mode is {normalize_mode!r}; '
                'use normalize_mode=ref_wells to apply reference-well bounds',
                stacklevel=2,
            )
    if img_type == 'multichannel' and normalize_mode != 'global':
        warnings.warn(
            f'multichannel plots use global normalization; overriding '
            f'{normalize_mode!r} with global',
            stacklevel=2,
        )
        return 'global'
    if img_type == 'mask':
        return 'off'
    return normalize_mode


def grayscale_display_limits(
        normalize_mode: NormalizeMode,
        global_pmin: np.ndarray,
        global_pmax: np.ndarray,
        channel_idx: int = 0,
    ) -> Tuple[float, float]:
    if normalize_mode == 'off':
        return float(global_pmin[channel_idx]), float(global_pmax[channel_idx])
    return 0.0, 1.0


def _accumulate_percentiles_from_df(
        df: pd.DataFrame,
        channels: List[int],
        flat_field: Optional[np.ndarray],
        pmin: float,
        pmax: float,
        path_col: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
    n_channels = len(channels)
    pmin_vals = np.zeros((n_channels,))
    pmax_vals = np.zeros((n_channels,))
    count = 0
    for _, row in df.iterrows():
        file = str(row[path_col])
        ref_img = read_tiff_or_nd2(file, bundle_axes='cyx').astype(float)
        if flat_field is not None:
            ref_img = flat_field_correction(ref_img, flat_field)
        for i, ch in enumerate(channels):
            plane = select_channel_plane(ref_img, ch)
            pmin_vals[i] += np.percentile(plane, pmin)
            pmax_vals[i] += np.percentile(plane, pmax)
        count += 1
    if count == 0:
        raise ValueError('No images found to compute percentiles')
    pmin_vals /= count
    pmax_vals /= count
    return pmin_vals, pmax_vals


def get_dataset_percentiles(
        df: pd.DataFrame,
        channels: Union[int, Sequence[int]],
        flat_field: Optional[np.ndarray],
        pmin: float = 0.1,
        pmax: float = 99.9,
        path_col: str = 'file',
    ) -> Tuple[np.ndarray, np.ndarray]:
    '''Mean per-image percentiles across all rows in ``df`` (global bounds).'''
    if path_col not in df.columns:
        raise ValueError(f'{path_col} not in df columns')
    channels = coerce_channels(channels)
    return _accumulate_percentiles_from_df(
        df, channels, flat_field, pmin, pmax, path_col
    )


def get_ref_wells_percentiles(
        df: pd.DataFrame,
        ref_wells: List[str],
        channels: Union[int, Sequence[int]],
        flat_field: Optional[np.ndarray],
        pmin: float = 0.1,
        pmax: float = 99.9,
        path_col: str = 'file',
        well_col: str = 'well',
    ) -> Tuple[np.ndarray, np.ndarray]:
    '''Percentile bounds from reference wells, per entry in ``channels``.'''
    if path_col not in df.columns:
        raise ValueError(f'{path_col} not in df columns')
    if well_col not in df.columns:
        raise ValueError(f'{well_col} not in df columns')

    channels = coerce_channels(channels)
    df_ref = df[df[well_col].isin(ref_wells)]
    if len(df_ref) == 0:
        raise ValueError('No reference wells found in DataFrame')
    return _accumulate_percentiles_from_df(
        df_ref, channels, flat_field, pmin, pmax, path_col
    )


def normalize_grayscale_plane(
        img: np.ndarray,
        normalize_mode: NormalizeMode,
        pmin: float,
        pmax: float,
        bounds_pmin: Optional[float] = None,
        bounds_pmax: Optional[float] = None,
) -> np.ndarray:
    if normalize_mode == 'off':
        return img
    if normalize_mode == 'per_image':
        return normalize_img(img, pmin=pmin, pmax=pmax, clip=True)
    return normalize_img(
        img,
        pmin_val=bounds_pmin,
        pmax_val=bounds_pmax,
        clip=True,
    )