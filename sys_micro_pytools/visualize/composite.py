import numpy as np
from matplotlib.colors import to_rgb

def create_composite2D(img, channel_dim, colors=None, normalize=True):
    img_shape = list(img.shape)
    n_channels = img_shape.pop(channel_dim)
    H,W = img_shape
    if colors is None:
        colors = ['c', 'm', 'y', 'g', 'w', 'r']
        colors = [to_rgb(c) for c in colors]
    n_colors = len(colors)
    if n_channels > n_colors:
        raise RuntimeError('The image has more than 6 channels for which there are default colors. ' +
            'Please provide a color to use for each channels')
    composite_img = np.zeros((H,W,3))
    for c in range(n_channels):
        channel = np.squeeze(img.take((c,), axis=channel_dim))
        channel2rgb = np.dstack((
        colors[c][0]*channel,
        colors[c][1]*channel,
        colors[c][2]*channel
        ))
        composite_img += channel2rgb
    for c in range(composite_img.shape[2]):
        composite_img[:,:,c] = composite_img[:,:,c] / (composite_img[:,:,c].max() + 1e-9)
    if normalize:
        composite_img = composite_img / (composite_img.max() + 1e-9)
    return composite_img
