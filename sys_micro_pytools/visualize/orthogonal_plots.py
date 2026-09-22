"""Module for creating orthogonal view plots of 3D multi-channel images."""

from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import tifffile
from matplotlib import pyplot as plt
from tqdm import tqdm


def create_orthogonal_plot(
    img: np.ndarray,
    channel_dim: Optional[int] = 1,
    cmaps: Optional[List[str]] = None,
    figsize_scale: float = 3.0,
    percentiles: Tuple[float, float] = (0.1, 99.9)
) -> plt.Figure:
    """Create a plot with orthogonal views (YX, YZ, XZ) through the middle of a 3D image.

    Args:
        img: 3D or 4D numpy array. For 4D, default dimension order is ZCYX.
        channel_dim: Dimension index for channels. Default is 1 (for ZCYX order).
            Set to None for 3D single-channel images.
        cmaps: List of colormap names for each channel. If None, uses 'gray' for all.
        figsize_scale: Scale factor for figure size. Default is 3.0.
        percentiles: Tuple of (pmin, pmax) percentiles for normalization. Default is (0.1, 99.9).

    Returns:
        matplotlib Figure with orthogonal views.
    """
    # Handle 3D single-channel case
    if img.ndim == 3:
        # Single channel image (ZYX)
        img = np.expand_dims(img, axis=1)  # Convert to ZCYX with C=1
        channel_dim = 1
    elif img.ndim != 4:
        raise ValueError(f"Expected 3D or 4D image, got {img.ndim}D")

    # Move channel dimension to position 1 if not already there
    if channel_dim != 1:
        img = np.moveaxis(img, channel_dim, 1)

    # Now image is in ZCYX order
    n_z, n_channels, n_y, n_x = img.shape

    # Normalize each channel using percentiles
    pmin, pmax = percentiles
    img_norm = img.astype(np.float32)
    for c in range(n_channels):
        channel_data = img_norm[:, c, :, :]
        vmin = np.percentile(channel_data, pmin)
        vmax = np.percentile(channel_data, pmax)
        if vmax > vmin:
            img_norm[:, c, :, :] = (channel_data - vmin) / (vmax - vmin)
        else:
            img_norm[:, c, :, :] = 0
    # Clip to 0-1 range
    img_norm = np.clip(img_norm, 0, 1)

    # Get middle indices for each dimension
    mid_z = n_z // 2
    mid_y = n_y // 2
    mid_x = n_x // 2

    # Extract orthogonal slices for each channel
    # YX plane (middle Z slice)
    yx_slices = img_norm[mid_z, :, :, :]  # Shape: (C, Y, X)
    # YZ plane (middle X slice)
    yz_slices = img_norm[:, :, :, mid_x]  # Shape: (Z, C, Y) -> need to transpose
    yz_slices = np.transpose(yz_slices, (1, 2, 0))  # Shape: (C, Y, Z)
    # XZ plane (middle Y slice)
    xz_slices = img_norm[:, :, mid_y, :]  # Shape: (Z, C, X) -> need to transpose
    xz_slices = np.transpose(xz_slices, (1, 0, 2))  # Shape: (C, Z, X)

    # Set up colormaps
    if cmaps is None:
        cmaps = ['gray'] * n_channels
    elif len(cmaps) < n_channels:
        # Extend with 'gray' if not enough colormaps provided
        cmaps = list(cmaps) + ['gray'] * (n_channels - len(cmaps))

    # Create figure with 3 rows (planes) and n_channels columns
    plane_labels = ['YX (Z-mid)', 'YZ (X-mid)', 'XZ (Y-mid)']
    slices = [yx_slices, yz_slices, xz_slices]

    # Calculate figure size based on actual slice dimensions
    max_width = max(yx_slices.shape[2], yz_slices.shape[2], xz_slices.shape[2])
    max_height = max(yx_slices.shape[1], yz_slices.shape[1], xz_slices.shape[1])
    aspect = max_width / max_height if max_height > 0 else 1

    fig_width = figsize_scale * n_channels * aspect
    fig_height = figsize_scale * 3

    fig, axes = plt.subplots(3, n_channels, figsize=(fig_width, fig_height), squeeze=False)

    for row, (plane_slices, label) in enumerate(zip(slices, plane_labels)):
        for col in range(n_channels):
            ax = axes[row, col]
            ax.imshow(plane_slices[col], cmap=cmaps[col], aspect='auto')
            ax.set_xticks([])
            ax.set_yticks([])

            # Set row labels on the left
            if col == 0:
                ax.set_ylabel(label, fontsize=10)

            # Set column labels on top
            if row == 0:
                ax.set_title(f'Channel {col}', fontsize=10)

    plt.tight_layout()
    return fig


def create_orthogonal_plots_batch(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    suffix: str = '.tif',
    channel_dim: Optional[int] = 1,
    cmaps: Optional[List[str]] = None,
    output_format: str = 'png',
    dpi: int = 150,
    figsize_scale: float = 3.0,
    percentiles: Tuple[float, float] = (0.1, 99.9)
) -> None:
    """Create orthogonal view plots for all images in a directory.

    Processes all images with the given suffix in input_path (including subdirectories),
    preserves the subfolder structure in output_path, and saves plots with the same
    filename (but different extension).

    Args:
        input_path: Path to directory containing images.
        output_path: Path to directory where plots will be saved.
        suffix: File extension of images to process. Default is '.tif'.
        channel_dim: Dimension index for channels. Default is 1 (for ZCYX order).
            Set to None for 3D single-channel images.
        cmaps: List of colormap names for each channel. If None, uses 'gray' for all.
        output_format: Format for output figures. Default is 'png'.
        dpi: DPI for output figures. Default is 150.
        figsize_scale: Scale factor for figure size. Default is 3.0.
        percentiles: Tuple of (pmin, pmax) percentiles for normalization. Default is (0.1, 99.9).
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Find all image files recursively, filtering out dot-prefixed files (Mac hidden files)
    files = sorted([
        f for f in input_path.rglob(f'*{suffix}')
        if f.is_file() and not f.name.startswith('.')
    ])

    if len(files) == 0:
        print(f"No files with suffix '{suffix}' found in {input_path}")
        return

    for file in tqdm(files, desc="Creating orthogonal plots"):
        # Read image
        img = tifffile.imread(str(file))

        # Create orthogonal plot
        fig = create_orthogonal_plot(
            img,
            channel_dim=channel_dim,
            cmaps=cmaps,
            figsize_scale=figsize_scale,
            percentiles=percentiles
        )

        # Determine output path preserving subfolder structure
        relative_path = file.relative_to(input_path)
        output_file = output_path / relative_path.with_suffix(f'.{output_format}')
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Save figure
        fig.savefig(output_file, dpi=dpi, bbox_inches='tight')
        plt.close(fig)

    print(f"Saved {len(files)} orthogonal plots to {output_path}")

