import numpy as np
from matplotlib.colors import to_rgb

from sys_micro_pytools.preprocess.normalize import normalize_img

def create_composite(img, channel_dim, colors=('cyan', 'green', 'orange', 'red', 'yellow', 'magenta' ), 
                     do_normalize=True, pmin=0, pmax=100, do_clip=True, vmin=0.0, vmax=1.0):
    """
    Create a composite image from a multi-channel image.
    
    Parameters:
        img (np.ndarray): The multi-channel image to composite.
        channel_dim (int): The dimension of the channels in the image.
        colors (list of str or tuple or np.ndarray): The colors to use for each channel.
        normalize (bool): Whether to normalize the image.
        pmin (float): The minimum percentile to normalize to.
        pmax (float): The maximum percentile to normalize to.
        clip (bool): Whether to clip the image.
        vmin (float): The minimum value to clip to.
        vmax (float): The maximum value to clip to.

    Returns:
        np.ndarray: The composite image with 3 channels.
    """
    img_shape = list(img.shape)
    n_channels = img_shape.pop(channel_dim)
    if all(isinstance(c, str) for c in colors):
        colors = [to_rgb(c) for c in colors]

    for i, color in enumerate(colors):
        if isinstance(color, str):
            color = to_rgb(color)
        elif isinstance(color, tuple):
            color = np.array(color)
        elif isinstance(color, np.ndarray):
            color = color
        else:
            raise ValueError(f'Invalid color type: {type(color)}. Must be str, tuple, or np.ndarray')
        if color.shape != (3,):
            raise ValueError(f'Invalid color shape: {color.shape}. Must be (3,)')
        colors[i] = color
        
    n_colors = len(colors)
    if n_channels > n_colors:
        raise RuntimeError('The image has more than 6 channels for which there are default colors. ' +
            'Please provide a color to use for each channels')
    
    # Preallocate composite image
    new_shape = img_shape.copy()
    new_shape.insert(channel_dim, 3)
    composite_img = np.zeros(new_shape)
    
    # Create composite image
    for i in range(n_channels):
        channel = img.take((i,), axis=channel_dim)
        channel2rgb = np.concatenate((
        colors[i][0]*channel,
        colors[i][1]*channel,
        colors[i][2]*channel
        ), axis=channel_dim)
        composite_img += channel2rgb
    if do_normalize:
        composite_img = normalize_img(composite_img, pmin=pmin, pmax=pmax, clip=False)
    if do_clip:
        composite_img = np.clip(composite_img, vmin, vmax)
    return composite_img
