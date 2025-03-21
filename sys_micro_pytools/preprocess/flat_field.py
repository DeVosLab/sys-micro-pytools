from argparse import ArgumentParser
from pathlib import Path
from typing import Union
from tqdm import tqdm
import tifffile
import numpy as np
from skimage.filters import gaussian

from sys_micro_pytools.io import read_tiff_or_nd2

def efficient_mean(current_mean, new_value, n):
    return current_mean + (new_value - current_mean) / n


def reduce_imgs(files, bundle_axes, method='mean'):
    assert method in ['mean', 'median'], 'Method must be either mean or median'

    if method == 'medain':
        imgs = []  # Collect all images for median calculation
    
    reduced_img = None
    for i, f in enumerate(tqdm(files)):
        img = read_tiff_or_nd2(str(f), bundle_axes=bundle_axes).astype(float)

        if method == 'medain':
            img = np.expand_dims(img, axis=0)
            imgs.append(img)
        else:
            if reduced_img is None:
                reduced_img = img  # Initialize reduced_img with the first image
            else:
                reduced_img = efficient_mean(reduced_img, img, i+1)  # Update the running mean
    
    if method == 'medain':
        imgs = np.concatenate(imgs, axis=0)
        reduced_img = np.median(imgs, axis=0)
    return reduced_img


def compute_flat_field(input_path, from_well=None, skip_wells=None, 
                       suffix='.nd2', n_images=None, sigma=90, 
                       grayscale=False, method='mean', seed=0):
    # Load images to compute flat field from
    files = sorted([f for f in input_path.iterdir() if f.is_file() and \
                    f.suffix == suffix and not f.stem.startswith('.')])
    if n_images is None:
        n_images = len(files)
    
    if from_well is not None:
        pattern = from_well
        files = [f for f in files if pattern in f.stem]
        type = 'pattern' # Only this pattern is used
    elif skip_wells is not None:
        pattern = skip_wells
        files = [f for f in files if pattern not in f.stem]
        type = 'batch' # All wells except this pattern are used
    else:
        np.random.seed(seed)
        files = np.random.choice(
            files,
            size=n_images,
            replace=False
        )
        type='batch' # Randomly selected images are used
    
    # Compute flat field per channel
    bundle_axes = 'cyx' if not grayscale else 'yx'
    flat_field = reduce_imgs(files, bundle_axes, method=method).astype(np.float32)
    
    # Gaussian Blur
    flat_field = gaussian(flat_field, sigma=sigma, channel_axis=0)

    return flat_field, type


def get_flat_field_files(input_path: Union[str, Path]):
    flat_field_path = Path(input_path)
    if flat_field_path.suffix == '.tif':
        flat_field_files = [flat_field_path]
    else:
        flat_field_files = sorted([f for f in flat_field_path.iterdir() if \
                                f.is_file() and f.suffix == '.tif' and not f.stem.startswith('.')])     
        flat_field_files = [f for f in flat_field_files if 'FF'in f.stem]  
    return flat_field_files


def flat_field_correction(img, flat_field):
    if img.shape == flat_field.shape:
        img /= flat_field
    elif flat_field.shape[0] == 1:
        for c in range(img.shape[0]):
            img[c,:,:] = img[c,:,:] / flat_field
    return img

def main(**kwargs):
    # Compute flat field
    input_path = Path(kwargs['input_path'])
    flat_field, type = compute_flat_field(
        input_path,
        from_well=kwargs['from_well'],
        skip_wells=kwargs['skip_wells'],
        suffix=kwargs['suffix'],
        n_images=kwargs['n_images'],
        sigma=kwargs['sigma'],
        grayscale=kwargs['grayscale'],
        method=kwargs['method'],
        seed=kwargs['seed']
    )

    # Save flat field
    output_path = Path(kwargs['output_path'])
    output_path.mkdir(parents=True, exist_ok=True)
    filename = f'FF_{input_path.stem}_{type}.tif'
    tifffile.imwrite(
        output_path.joinpath(filename),
        flat_field,
        compression='zlib',
        imagej=True,
        metadata={'axes': 'CYX' if not kwargs['grayscale'] else 'YX'}
    )
    

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('-i', '--input_path', type=Path, required=True,
        help='Path to the input images')
    parser.add_argument('-o', '--output_path', type=Path, required=True,
        help='Path to save the flat field')
    parser.add_argument('--suffix', type=str, default='.nd2',
        help='Suffix of the input images')
    parser.add_argument('--sigma', type=float, default=90,
        help='Sigma of the Gaussian blur filter')
    parser.add_argument('--n_images', type=int, default=None,
        help='Number of images to use to generate the flat field')
    parser.add_argument('--from_well', type=str, default=None,
        help=('Well to use to generate the flat field.' 
              'If None, use n random images from the '
              'input path to generate the flat field'))
    parser.add_argument('--skip_wells', type=str, default=None,
        help=('Wells to ignore when generating the flat field.'
        'If None, use all wells.'))
    parser.add_argument('--grayscale', action='store_true',
        help='Use grayscale images instead of RGB images.')
    parser.add_argument('--method', type=str, default='mean', 
                        choices=['mean', 'median'],
        help=('Method to use to generate the flat field.'
              'Otherwise, use mean.'))
    parser.add_argument('--seed', type=int, default=0,
        help='Seed for random number generator')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    main(**vars(args))
