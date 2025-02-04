from typing import Union, Tuple
import numpy as np
from skimage.segmentation import clear_border
from skimage.filters import threshold_otsu
from skimage.registration import phase_cross_correlation
from scipy.ndimage import shift
from sys_micro_pytools.preprocess.dimensionality import unsqueeze_to_ndim, squeeze_to_ndim

def crop_center(img: np.ndarray, new_size: Union[Tuple[int, int], Tuple[int, int, int]]) -> np.ndarray:
    '''Crop the center of an image to a new size.

    Parameters
    ----------
    img : ndarray
        Image to crop with shape (H, W), (C, H, W), (D, H, W) or (D, C, H, W).
    new_size : tuple
        New size to crop to with shape (H, W) or (D, H, W).

    Returns
    -------
    img : ndarray
        Cropped image with shape (H, W), (C, H, W), (D, H, W) or (D, C, H, W).
    '''
    # Store the original ndim
    ndim = img.ndim

    # Unsqueeze the image and new_size to 4D
    img = unsqueeze_to_ndim(img, 4) # Z, C, H, W
    new_size = unsqueeze_to_ndim(np.array(new_size), 3) # Z, H, W

    # Get the original and new sizes
    D, _, H, W = img.shape
    D_new, H_new, W_new = new_size.shape

    # Check if the new size is larger than the original size
    if D_new > D or H_new > H or W_new > W:
        raise ValueError(f'new_size is larger than img, found img.shape={img.shape} and new_size={new_size}')
    
    # Calculate the start indices for the crop
    # If the new size is 1, don't crop
    startz = D//2-(D_new//2) if D_new > 1 else 0
    starty = H//2-(H_new//2) if H_new > 1 else 0
    startx = W//2-(W_new//2) if W_new > 1 else 0

    # Crop the image
    img = img[startz:startz+D_new,:,starty:starty+H_new,startx:startx+W_new]

    # Squeeze the image back to the original ndim and return
    return squeeze_to_ndim(img, ndim)

def align_imgs(img_fixed, img_moving, channel2use=0, cval=0, T=None):
    '''Align two images using phase cross correlation.

    Parameters
    ----------
    img_fixed : ndarray
        Fixed image with shape (C, H, W) or (D, C, H, W).
    img_moving : ndarray
        Moving image with shape (C, H, W) or (D, C, H, W).
    channel2use : int
        Channel to use for alignment.
    cval : int
        Value to use for padding.
    T : array_like
        Translation vector with shape (D, 2) or (D, 3).

    Returns
    -------
    img_moving : ndarray
        Aligned moving image with shape (C, H, W) or (D, C, H, W).
    T : array_like
        Translation vector with shape (D, 2) or (D, 3).
    '''
    ndim = img_fixed.ndim
    img_fixed = unsqueeze_to_ndim(img_fixed, 4)
    img_moving = unsqueeze_to_ndim(img_moving, 4)

    if T is None:
        T, _, _ = phase_cross_correlation(
            img_fixed[...,channel2use,:,:],
            img_moving[...,channel2use,:,:],
            normalization='phase'
            )
    T = np.insert(T, values=0, axis=1) # add 0 for channel dimension
    img_moving = shift(img_moving, T, cval=cval)
    img_moving = squeeze_to_ndim(img_moving, ndim)
    T = squeeze_to_ndim(T, ndim-1)
    return img_moving, T


def align_GT2CP(img_CP, masks, img_GT, channels2use_CP=0, channels2use_GT=0, 
                new_size=(2048,2048), labels=None):
    '''Align the GT image to the CP image using phase cross correlation.

    Parameters
    ----------
    img_CP : ndarray
        CP image with shape (C, H, W) or (D, C, H, W).
    masks : ndarray
        Masks with shape (H, W) or (D, H, W).
    img_GT : ndarray
        GT image with shape (C, H, W) or (D, C, H, W).
    channels2use_CP : int
        Channel to use for alignment.
    channels2use_GT : int
        Channel to use for alignment.
    new_size : tuple
        New size to crop to with shape (H, W) or (D, H, W).
    labels : list
        Labels to use for alignment (optional).

    Returns
    -------
    img_GT_moved : ndarray
        Aligned GT image with shape (C, H, W) or (D, C, H, W).
    labels : list
        Labels that were used for alignment.
    '''

    # Get the labels from the masks. If labels is defined, only keep the labels that are in labels.
    if labels is None:
        labels = np.unique(masks)
        labels = list(labels[labels != 0])
    else:
        labels_present = np.unique(masks)
        labels_present = list(labels_present[labels_present != 0])
        for lbl in labels_present:
            if lbl not in labels:
                masks = np.where(masks == lbl, 0, masks)
    # Unsqueeze the images and masks to 3D/4D (D, C, H, W)
    ndim_CP = img_CP.ndim
    ndim_GT = img_GT.ndim
    ndim_masks = masks.ndim
    img_CP = unsqueeze_to_ndim(img_CP, 4)   # D, C, H, W
    img_GT = unsqueeze_to_ndim(img_GT, 4)   # D, C, H, W
    masks = unsqueeze_to_ndim(masks, 3)     # D, H, W
    new_size = unsqueeze_to_ndim(np.array(new_size), 3) # D, H, W

    # Get the max over all channels for the GT and CP images
    img_GT_max_channels2use = img_GT[:,np.r_[channels2use_GT],:,:].max(axis=1) # D, H, W
    img_CP_max_channels2use = img_CP[:,np.r_[channels2use_CP],:,:].max(axis=1) # D, H, W
    
    # Calculate the cross-correlation between the GT and CP images
    image_product = np.fft.fftn(img_GT_max_channels2use) * np.fft.fft2(img_CP_max_channels2use).conj()
    cc_image = np.fft.fftshift(np.fft.ifftn(image_product)).real # D, H, W

    # Find the shift that maximizes the cross-correlation
    ind = np.unravel_index(np.argmax(cc_image, axis=None), cc_image.shape) # (z, y, x)
    D, _, H, W = img_CP.shape
    T = np.array(ind) - np.array([D/2, H/2, W/2]) # (z, y, x)

    # Align the images
    T = np.insert(T, values=0, axis=1) # add 0 for channel dimension
    img_GT_moved, _ = align_imgs(img_CP, img_GT, T=-T) # D, C, H, W
    img_GT_max_channels2use_moved = img_GT_moved[:,np.r_[channels2use_GT],:,:].max(axis=1) # D, H, W

    # Crop out the center "new_size" region
    img_CP = crop_center(img_CP, new_size) # D_new, C, H_new, W_new
    img_GT_moved = crop_center(img_GT_moved, new_size) # D_new, C, H_new, W_new
    masks = crop_center(masks, new_size) # D_new, H_new, W_new
    masks = squeeze_to_ndim(masks, ndim_masks) # Needed for clear border, otherwise all masks could be cleared
    masks = clear_border(masks)
    masks = unsqueeze_to_ndim(masks, 3)

    # Identify incomplete masks in img_GT_moved and remove them
    missing_GT_data = shift(np.zeros_like(img_GT_max_channels2use), shift=T, cval=1).astype(bool) # Binary mask being 1 where data is missing
    missing_GT_data = crop_center(missing_GT_data, new_size)#[0,]
    masks_missing = np.where(missing_GT_data, masks, 0)
    labels_missing = np.unique(masks_missing)
    labels_missing = list(labels_missing[labels_missing != 0])
    labels = [lbl for lbl in labels if lbl not in labels_missing]
    masks = np.where(np.isin(masks, labels), masks, 0)

    # Calculate ratio of foreground area in CP and GT images
    # for each mask to check if cells are present in both images
    img_CP_max_channels2use = crop_center(img_CP_max_channels2use, new_size) # D_new, H_new, W_new
    img_GT_max_channels2use_moved = crop_center(img_GT_max_channels2use_moved, new_size) # D_new, H_new, W_new
    binary_mask = np.where(masks > 0, True, False)
    binary_CP = (img_CP_max_channels2use > threshold_otsu(img_CP_max_channels2use)) * binary_mask
    binary_GT = (img_GT_max_channels2use_moved > threshold_otsu(img_GT_max_channels2use)) * binary_mask # Use original GT threshold to avoid fill values to be influencing the threshold
    for lbl in labels[:]:
        ratio = binary_GT[masks == lbl].sum()/binary_CP[masks == lbl].sum()
        if ratio < 0.2:
            masks = np.where(masks == lbl, 0, masks)
            labels.remove(lbl)

    # Squeeze masks and img_GT to original ndim
    img_CP = squeeze_to_ndim(img_CP, ndim_CP)
    masks = squeeze_to_ndim(masks, ndim_masks)
    img_GT_moved = squeeze_to_ndim(img_GT_moved, ndim_GT)

    return img_CP, masks, img_GT_moved, labels