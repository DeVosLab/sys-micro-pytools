"""Volume rendering utilities for 3D image visualization using PyVista."""

import numpy as np
import pyvista as pv
from typing import List, Tuple, Optional, Union, Dict
from dataclasses import dataclass

@dataclass
class VolumeSettings:
    """Settings for volume rendering."""
    background_color: Union[str, Tuple[float, float, float]] = 'black'
    lighting: str = 'three lights'
    show_axes: bool = True
    axes_labels: bool = True
    camera_position: str = 'isometric'
    
def create_multichannel_volume(
    images: List[np.ndarray],
    channel_names: List[str],
    channel_colors: List[str],
    voxel_size: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    opacity_mode: str = 'linear',
    volume_settings: Optional[VolumeSettings] = None
) -> pv.Plotter:
    """Create a multi-channel volume rendering using PyVista.
    
    Parameters
    ----------
    images : List[np.ndarray]
        List of 3D numpy arrays, each representing a channel. Shape should be (Z, Y, X).
    channel_names : List[str]
        Names for each channel.
    channel_colors : List[str]
        Colors for each channel (e.g., ['cyan', 'red', 'green']).
    voxel_size : Tuple[float, float, float]
        Voxel size in (Z, Y, X) order.
    opacity_mode : str
        Opacity mapping mode ('linear', 'sigmoid', or custom opacity array).
    volume_settings : Optional[VolumeSettings]
        Volume rendering settings.
        
    Returns
    -------
    pv.Plotter
        PyVista plotter object ready for display.
    """
    if volume_settings is None:
        volume_settings = VolumeSettings()
        
    if len(images) != len(channel_names) or len(images) != len(channel_colors):
        raise ValueError("Number of images, names, and colors must match")
    
    # Create plotter
    plotter = pv.Plotter(lighting=volume_settings.lighting)
    plotter.set_background(volume_settings.background_color)
    
    # Process each channel
    for i, (img, name, color) in enumerate(zip(images, channel_names, channel_colors)):
        # Convert to PyVista ImageData
        dims = img.shape[::-1]  # Convert (Z,Y,X) to (X,Y,Z)
        spacing = voxel_size[::-1]  # Convert (Z,Y,X) to (X,Y,Z)
        
        grid = pv.ImageData(dimensions=dims, spacing=spacing)
        grid.point_data['values'] = img.flatten(order='F')
        
        # Create color map based on channel color
        cmap = _create_channel_colormap(color)
        
        # Add volume to plotter
        plotter.add_volume(
            grid,
            cmap=cmap,
            opacity=opacity_mode,
            name=name,
            show_scalar_bar=False
        )
    
    # Add axes if requested
    if volume_settings.show_axes:
        plotter.add_axes(
            xlabel='X (μm)',
            ylabel='Y (μm)',
            zlabel='Z (μm)',
            line_width=5,
            labels_off=not volume_settings.axes_labels
        )
    
    # Set camera position
    plotter.camera_position = volume_settings.camera_position
    
    # Add channel info text
    channel_info = ", ".join([f"{name} ({color})" for name, color in zip(channel_names, channel_colors)])
    plotter.add_text(
        f"Channels: {channel_info}",
        position='upper_left',
        font_size=12,
        color='white'
    )
    
    return plotter

def create_isosurface_multichannel(
    images: List[np.ndarray],
    channel_names: List[str], 
    channel_colors: List[str],
    thresholds: List[float],
    voxel_size: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    opacity: float = 0.7,
    smooth_iterations: int = 100,
    volume_settings: Optional[VolumeSettings] = None
) -> pv.Plotter:
    """Create isosurface rendering for multiple channels.
    
    Parameters
    ----------
    images : List[np.ndarray]
        List of 3D numpy arrays, each representing a channel.
    channel_names : List[str]
        Names for each channel.
    channel_colors : List[str]  
        Colors for each channel.
    thresholds : List[float]
        Isosurface threshold for each channel.
    voxel_size : Tuple[float, float, float]
        Voxel size in (Z, Y, X) order.
    opacity : float
        Surface opacity.
    smooth_iterations : int
        Number of smoothing iterations for surfaces.
    volume_settings : Optional[VolumeSettings]
        Volume rendering settings.
        
    Returns
    -------
    pv.Plotter
        PyVista plotter object ready for display.
    """
    if volume_settings is None:
        volume_settings = VolumeSettings()
        
    if len(images) != len(channel_names) or len(images) != len(channel_colors) or len(images) != len(thresholds):
        raise ValueError("Number of images, names, colors, and thresholds must match")
    
    # Create plotter
    plotter = pv.Plotter(lighting=volume_settings.lighting)
    plotter.set_background(volume_settings.background_color)
    
    # Process each channel
    for img, name, color, threshold in zip(images, channel_names, channel_colors, thresholds):
        # Convert to PyVista ImageData
        dims = img.shape[::-1]  # Convert (Z,Y,X) to (X,Y,Z)
        spacing = voxel_size[::-1]  # Convert (Z,Y,X) to (X,Y,Z)
        
        grid = pv.ImageData(dimensions=dims, spacing=spacing)
        grid.point_data['values'] = img.flatten(order='F')
        
        # Create isosurface
        isosurface = grid.contour([threshold])
        
        # Smooth the surface
        if smooth_iterations > 0:
            isosurface.smooth(
                n_iter=smooth_iterations,
                relaxation_factor=0.01,
                inplace=True
            )
        
        # Add to plotter
        plotter.add_mesh(
            isosurface,
            color=color,
            opacity=opacity,
            name=name,
            pbr=True,
            metallic=0.8,
            diffuse=1.0,
            specular=0.5
        )
    
    # Add axes if requested
    if volume_settings.show_axes:
        plotter.add_axes(line_width=5, labels_off=not volume_settings.axes_labels)
    
    # Set camera position
    plotter.camera_position = volume_settings.camera_position
    
    return plotter

def _create_channel_colormap(color: str) -> List[Tuple[float, float, float, float]]:
    """Create a colormap from transparent to the specified color.
    
    Parameters
    ----------
    color : str
        Color name (e.g., 'cyan', 'red', 'green', 'blue', 'yellow', 'magenta').
        
    Returns
    -------
    List[Tuple[float, float, float, float]]
        RGBA colormap from transparent to the specified color.
    """
    color_map = {
        'cyan': (0, 1, 1),
        'red': (1, 0, 0),
        'green': (0, 1, 0),
        'blue': (0, 0, 1),
        'yellow': (1, 1, 0),
        'magenta': (1, 0, 1),
        'white': (1, 1, 1),
        'gray': (0.5, 0.5, 0.5),
        'orange': (1, 0.5, 0),
        'purple': (0.5, 0, 1)
    }
    
    color_lower = color.lower()
    if color_lower in color_map:
        r, g, b = color_map[color_lower]
        return [(0, 0, 0, 0), (r, g, b, 1)]  # Transparent to color
    else:
        # Default to white if color not recognized
        return [(0, 0, 0, 0), (1, 1, 1, 1)]

def render_volume_with_slices(
    image: np.ndarray,
    voxel_size: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    slice_positions: Optional[Dict[str, int]] = None,
    volume_settings: Optional[VolumeSettings] = None
) -> pv.Plotter:
    """Render a 3D volume with optional orthogonal slice planes.
    
    Parameters
    ----------
    image : np.ndarray
        3D image array with shape (Z, Y, X).
    voxel_size : Tuple[float, float, float]
        Voxel size in (Z, Y, X) order.
    slice_positions : Optional[Dict[str, int]]
        Dictionary with slice positions for each axis: {'x': pos, 'y': pos, 'z': pos}.
        If None, slices will be positioned at the center of each axis.
    volume_settings : Optional[VolumeSettings]
        Volume rendering settings.
        
    Returns
    -------
    pv.Plotter
        PyVista plotter object ready for display.
    """
    if volume_settings is None:
        volume_settings = VolumeSettings()
        
    # Create plotter
    plotter = pv.Plotter(lighting=volume_settings.lighting)
    plotter.set_background(volume_settings.background_color)
    
    # Convert to PyVista ImageData
    dims = image.shape[::-1]  # Convert (Z,Y,X) to (X,Y,Z)
    spacing = voxel_size[::-1]  # Convert (Z,Y,X) to (X,Y,Z)
    
    grid = pv.ImageData(dimensions=dims, spacing=spacing)
    grid.point_data['values'] = image.flatten(order='F')
    
    # Add volume rendering
    plotter.add_volume(grid, opacity='linear', cmap='viridis')
    
    # Add slice planes if requested
    if slice_positions is not None:
        z_size, y_size, x_size = image.shape
        
        # X slice (YZ plane)
        if 'x' in slice_positions:
            x_pos = slice_positions['x'] * voxel_size[2]  # Convert to physical coordinates
            x_slice = grid.slice(normal='x', origin=(x_pos, 0, 0))
            plotter.add_mesh(x_slice, cmap='viridis', opacity=0.8)
            
        # Y slice (XZ plane)  
        if 'y' in slice_positions:
            y_pos = slice_positions['y'] * voxel_size[1]
            y_slice = grid.slice(normal='y', origin=(0, y_pos, 0))
            plotter.add_mesh(y_slice, cmap='viridis', opacity=0.8)
            
        # Z slice (XY plane)
        if 'z' in slice_positions:
            z_pos = slice_positions['z'] * voxel_size[0]
            z_slice = grid.slice(normal='z', origin=(0, 0, z_pos))
            plotter.add_mesh(z_slice, cmap='viridis', opacity=0.8)
    
    # Add axes if requested
    if volume_settings.show_axes:
        plotter.add_axes(
            xlabel='X (μm)',
            ylabel='Y (μm)',
            zlabel='Z (μm)',
            line_width=5,
            labels_off=not volume_settings.axes_labels
        )
    
    # Set camera position
    plotter.camera_position = volume_settings.camera_position
    
    return plotter 