from pathlib import Path
from nd2reader import ND2Reader
import tifffile


def read_nd2(filename, bundle_axes='cyx'):
    with ND2Reader(filename) as images:
        if bundle_axes is not None:
            images.bundle_axes = bundle_axes
            images = images[0]
        return images
    
def read_tif_or_nd2(filename, bundle_axes: str = 'cyx'):
    filename = Path(filename)
    if filename.suffix == '.tif':
        img = tifffile.imread(str(filename))
    elif filename.suffix == '.nd2':
        img = read_nd2(str(filename), bundle_axes)
    return img