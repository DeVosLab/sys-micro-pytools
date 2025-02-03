import tifffile
import numpy as np
import pandas as pd
from typing import Union, List, Tuple, Optional, Dict, Callable
from skimage.measure import label, regionprops_table
from skimage.filters import threshold_otsu
from skimage.transform import rescale
from skimage.morphology import (
    remove_small_objects,
    binary_opening,
    binary_closing,
    binary_dilation,
    binary_erosion,
    ball,
)
from skimage.segmentation import clear_border, watershed
from scipy.ndimage import median_filter, binary_fill_holes, distance_transform_edt
import pyvista as pv
from dataclasses import dataclass
import colorsys
import time


@dataclass
class SurfaceAttributes:
    color: str = 'white'  # default color
    opacity: float = 1.0
    pbr: bool = True
    metallic: float = 0.8
    diffuse: float = 1.0
    specular: float = 0.5
    
    def __init__(self, **kwargs):
        # Set default values first
        self.color = 'white'
        self.opacity = 1.0
        self.pbr = True
        self.metallic = 0.8
        self.diffuse = 1.0
        self.specular = 0.5
        
        # Update with any provided values and add new attributes
        for key, value in kwargs.items():
            setattr(self, key, value)

@dataclass
class RenderSettings:
    background_color: Union[list, str] = (0.9, 0.9, 0.9)
    lighting: str = 'three lights'
    show_axes: bool = True
    axes_line_width: int = 5
    axes_labels: bool = False

def generate_distinct_colors(n: int) -> List[Tuple[float, float, float]]:
    """Generate n visually distinct colors using HSV color space.
    
    Parameters
    ----------
    n : int
        Number of colors to generate
    
    Returns
    -------
    List[Tuple[float, float, float]]
        List of RGB colors, each component in range [0,1]
    """
    colors = []
    for i in range(n):
        hue = i / n
        # Limit saturation and value to ensure valid RGB values
        saturation = min(max(0.7 + np.random.uniform(-0.2, 0.2), 0.0), 1.0)
        value = min(max(0.9 + np.random.uniform(-0.2, 0.2), 0.0), 1.0)
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        colors.append(rgb)
    return colors

def render_surfaces(
    surfaces: Dict[int, pv.PolyData],
    render_settings: Optional[RenderSettings] = None,
    surface_attributes: Optional[Dict[int, SurfaceAttributes]] = None,
    opacity: Optional[float] = None
) -> None:
    """Render PyVista surfaces using PyVista Plotter with flexible settings.

    Parameters
    ----------
    surfaces : Dict[int, pv.PolyData]
        Dictionary mapping label IDs to PyVista surfaces
    render_settings : Optional[RenderSettings]
        Global rendering settings
    surface_attributes : Optional[Dict[int, SurfaceAttributes]]
        Dictionary mapping label IDs to surface-specific rendering attributes.
        If a label is not in the dict, a distinct color will be assigned.
    opacity : Optional[float]
        Opacity of the surfaces. If provided, this overrides the opacity in surface_attributes.
    """
    # Use default settings if none provided
    if render_settings is None:
        render_settings = RenderSettings()
    
    # Initialize plotter with global settings
    plotter = pv.Plotter(lighting=render_settings.lighting)
    plotter.set_background(color=render_settings.background_color)
    
    # Generate distinct colors for labels without specific attributes
    labels_without_attrs = [label for label in surfaces.keys() 
                          if not surface_attributes or label not in surface_attributes]
    distinct_colors = generate_distinct_colors(len(labels_without_attrs))
    color_mapping = dict(zip(labels_without_attrs, distinct_colors))
    
    # Create default surface attributes (without color)
    default_attributes = SurfaceAttributes()
    
    # Add each surface with its specific or default attributes
    for label, surface in surfaces.items():
        if surface_attributes and label in surface_attributes:
            # Use specified attributes
            attrs = surface_attributes[label]
        else:
            # Use default attributes but with a distinct color
            attrs = default_attributes
            attrs.color = color_mapping[label]
        
        if opacity is not None:
            attrs.opacity = opacity
        
        plotter.add_mesh(
            surface,
            color=attrs.color,
            opacity=attrs.opacity,
            pbr=attrs.pbr,
            metallic=attrs.metallic,
            diffuse=attrs.diffuse,
            specular=attrs.specular
        )
    
    if render_settings.show_axes:
        plotter.add_axes(
            line_width=render_settings.axes_line_width,
            labels_off=not render_settings.axes_labels
        )
    
    plotter.show()

    return plotter


def labels2surface3D(masks: Union[np.ndarray, List[np.ndarray]], scale: tuple=(1, 1, 1), 
                     labels: Optional[Union[int, List[int], Tuple[int]]]=None,
                     smooth_iter: int=100, relaxation_factor: float=0.01, 
                     boundary_smoothing: bool=False) -> Dict[int, pv.PolyData]:
    ''' Create a PyVista surface from a label image

    Parameters
    ----------
    masks : Union[np.ndarray, List[np.ndarray]]
        If a single np.ndarray, labeled masks with values indicating different objects. 
        If a list of np.ndarray, each np.ndarray is a boolean mask for a separate object.
    scale : tuple
        Scale of the image
    labels : Optional[Union[int, List[int], Tuple[int]]]
        Labels to include in the surface. If None, labels will be 1, 2, 3, etc.
        If providing a list of masks, each mask is treated as a separate labeled object.
    smooth_iter : int
        Number of smoothing iterations
    relaxation_factor : float
        Relaxation factor for smoothing
    boundary_smoothing : bool
        Whether to smooth the boundary

    Returns
    -------
    surfaces : Dict[int, pv.PolyData]
        PyVista surfaces for each label
    '''

    if isinstance(masks, np.ndarray):
        labels_found = np.unique(masks)
        n_labels = len(labels_found[labels_found != 0])
    elif isinstance(masks, list):
        n_labels = len(masks)
    else:
        raise ValueError(f"Invalid masks type: {type(masks)}")
    
    if labels is None:
        labels = np.arange(1, n_labels + 1)
    elif isinstance(labels, int):
        labels = np.array([labels])
    elif isinstance(labels, (list, tuple)):
        labels = np.array(labels)

    if len(labels) != n_labels:
        raise ValueError(f"Number of labels ({len(labels)}) does not match number of masks ({n_labels})")

    surfaces = {}
    for i, lbl in enumerate(labels):
        if isinstance(masks, np.ndarray):
            mask = masks == lbl
        elif isinstance(masks, list):
            mask = masks[i]
        
        grid = pv.ImageData()
        grid.dimensions = np.array(mask.shape) + 1
        grid.spacing = scale
        grid.point_data["values"] = np.pad(mask, ((0,1), (0,1), (0,1))).flatten(order="F")
        surface = grid.contour([0.5])
        surface.smooth(
            n_iter=smooth_iter,
            boundary_smoothing=boundary_smoothing,
            relaxation_factor=relaxation_factor,
            inplace=True
        )
        surfaces[lbl] = surface
    return surfaces

def animate_surfaces(
    surfaces: Dict[int, pv.PolyData],
    transformations: Dict[int, List[Callable]],
    n_frames: int = 100,
    render_settings: Optional[RenderSettings] = None,
    surface_attributes: Optional[Dict[int, SurfaceAttributes]] = None,
    opacity: Optional[float] = None,
    output_path: str = "animation.gif",
    camera_position: str = "3d",
    bounds: Optional[Tuple[float, float, float, float, float, float]] = None
) -> None:
    """Animate surfaces with specified transformations."""
    if render_settings is None:
        render_settings = RenderSettings()

    # Initialize plotter with global settings
    plotter = pv.Plotter(notebook=False, off_screen=True)
    plotter.set_background(color=render_settings.background_color)
    
    # Store actors for each surface
    actors = {}
    
    # Initial setup of actors
    default_attributes = SurfaceAttributes()
    color_mapping = generate_distinct_colors(len(surfaces))
    for label, surface in surfaces.items():
        if surface_attributes and label in surface_attributes:
            attrs = surface_attributes[label]
        else:
            attrs = default_attributes
            attrs.color = color_mapping[label]
        
        if opacity is not None:
            attrs.opacity = opacity
        
        # Store the actor reference
        actors[label] = plotter.add_mesh(
            surface,
            color=attrs.color,
            opacity=attrs.opacity,
            pbr=attrs.pbr,
            metallic=attrs.metallic,
            diffuse=attrs.diffuse,
            specular=attrs.specular
        )
    
    if render_settings.show_axes:
        plotter.add_axes(
            line_width=render_settings.axes_line_width,
            labels_off=not render_settings.axes_labels
        )
    
    # Set bounds
    if bounds is not None:
        plotter.add_mesh(pv.Box(bounds=bounds), opacity=0, color='white')
    
    # Set camera position
    plotter.camera_position = camera_position

    # Open a gif
    plotter.open_gif(output_path)

    # Create animation frames
    for frame in range(n_frames):
        # Update each surface according to its transformations
        for label, surface in surfaces.items():
            if label in transformations:
                # Apply transformations for this label
                current_surface = surface.copy()
                if surface_attributes and label in surface_attributes:
                    current_attrs = surface_attributes[label]
                else:
                    current_attrs = default_attributes
                    current_attrs.color = color_mapping[label]
                for transform_func in transformations[label]:
                    result = transform_func(current_surface, frame, n_frames)
                    if isinstance(result, tuple):
                        current_surface, transform_attr = result
                        for key, value in transform_attr.items():
                            setattr(current_attrs, key, value)
                    else:
                        current_surface = result

                # Remove old actor and add new one
                plotter.remove_actor(actors[label])
                
                if opacity is not None:
                    current_attrs.opacity = opacity
                
                actors[label] = plotter.add_mesh(
                    current_surface,
                    **{attr: getattr(current_attrs, attr) for attr in vars(current_attrs)}
                )
        
        # Write the frame
        plotter.write_frame()

    # Close and finalize the animation
    plotter.close()

# Example transformation functions
def rotate(axis: Union[np.ndarray, List[float]], degrees: float = 360, 
          center: Union[np.ndarray, List[float], int]=None, speed_factor: float=1.0,
          surfaces: Dict[int, pv.PolyData]=None):
    """Create a rotation transformation with nonlinear speed control.
    
    Parameters
    ----------
    axis : Union[np.ndarray, List[float]]
        Axis of rotation
    degrees : float
        Total angle to rotate in degrees
    center : Union[np.ndarray, List[float], int]
        Center of rotation. Can be:
        - None: uses object's own center
        - np.ndarray/List: specific coordinates
        - int: ID of surface to use as center of rotation
    speed_factor : float
        Controls the nonlinearity of rotation speed:
        - speed_factor = 1.0: linear (constant) rotation speed
        - speed_factor < 1.0: rotation slows down over time
        - speed_factor > 1.0: rotation accelerates over time
    surfaces : Dict[int, pv.PolyData]
        Dictionary of all surfaces, required if center is specified as surface ID
    """
    axis = np.array(axis) / np.linalg.norm(axis)
    if isinstance(center, int):
        reference_id = center
        fixed_center = None
        initial_displacement = np.array(surfaces[reference_id].center) - np.array(center)
    else:
        reference_id = None
        fixed_center = center
        initial_displacement = None
    
    
    def transform(surface: pv.PolyData, frame: int, total_frames: int) -> pv.PolyData:
        # Get the center of rotation
        if reference_id is not None:
            # Use the current position of the reference surface
            rot_center = np.array(surface.center) - initial_displacement
        elif fixed_center is not None:
            rot_center = np.array(fixed_center)
        else:
            rot_center = np.array(surface.center)

        print(f'Frame {frame}: rot_center: {rot_center}')

        # Create a copy of the surface
        transformed = surface.copy()
        
        # Move to origin
        translation = -rot_center
        transformed.translate(translation, inplace=True)
        
        # Calculate nonlinear progress
        linear_progress = frame / total_frames
        nonlinear_progress = linear_progress ** speed_factor
        current_angle = degrees * nonlinear_progress
        
        # Rotate
        transformed.rotate_vector(axis, current_angle, inplace=True)
        
        # Move back to original position
        translation = rot_center
        transformed.translate(translation, inplace=True)
        
        return transformed
    return transform

def translate(direction: Union[np.ndarray, List[float]], distance: float = 10, speed_factor: float = 1.0):
    """Create a translation transformation with nonlinear speed control.
    
    Parameters
    ----------
    direction : Union[np.ndarray, List[float]]
        Direction vector for translation
    distance : float
        Total distance to translate
    speed_factor : float
        Controls the nonlinearity of translation speed:
        - speed_factor = 1.0: linear (constant) translation speed
        - speed_factor < 1.0: translation slows down over time
        - speed_factor > 1.0: translation accelerates over time
    """
    direction = np.array(direction) / np.linalg.norm(direction)
    def transform(surface: pv.PolyData, frame: int, total_frames: int) -> pv.PolyData:
        # Calculate nonlinear progress
        linear_progress = frame / total_frames
        nonlinear_progress = linear_progress ** speed_factor
        current_distance = distance * nonlinear_progress
        return surface.translate(direction * current_distance, inplace=True)
    return transform

def set_center(center: Union[np.ndarray, List[float]]):
    """ Set the center of the surface """
    def transform(surface: pv.PolyData, frame: int=None, total_frames: int=None) -> pv.PolyData:
        return surface.translate(-np.array(center), inplace=True)
    return transform

def wait():
    """Create a transformation that does nothing, useful for pausing between other transformations.
    
    Returns
    -------
    Callable
        A transformation function that returns the surface unchanged
    """
    def transform(surface: pv.PolyData, frame: int, total_frames: int) -> pv.PolyData:
        return surface
    return transform

def hide():
    """Create a transformation that immediately makes an object invisible.
    
    Returns
    -------
    Callable
        A transformation function that makes the surface invisible
    """
    def transform(surface: pv.PolyData, frame: int, total_frames: int) -> pv.PolyData:
        # Create opacity array of zeros
        transform_attr = {"opacity": 0}
        return surface, transform_attr
    return transform

def show(opacity: float=1):
    """Create a transformation that makes an object visible.
    
    Returns
    -------
    Callable
        A transformation function that makes the surface visible
    """
    def transform(surface: pv.PolyData, frame: int, total_frames: int) -> pv.PolyData:
        return surface, {"opacity": opacity}
    return transform

def change_color(target_color: Union[str, Tuple[float, float, float]]):
    """Create a transformation that changes the color of an object.
    
    Parameters
    ----------
    target_color : Union[str, Tuple[float, float, float]]
        The target color to transition to. Can be a string name or RGB tuple.
    
    Returns
    -------
    Callable
        A transformation function that changes the surface color
    """
    def transform(surface: pv.PolyData, frame: int, total_frames: int) -> pv.PolyData:
        return surface, {"color": target_color}
    return transform

@dataclass
class Transform:
    func: Callable
    start_frame: int
    end_frame: int

class SequentialTransform:
    def __init__(self, *transforms: Union[Callable, List[Callable]], frame_fractions: Optional[List[float]] = None):
        """
        Parameters
        ----------
        transforms : Union[Callable, List[Callable]]
            Transform functions to apply sequentially. Can be single transforms or lists of
            transforms to be applied simultaneously.
        frame_fractions : Optional[List[float]]
            List of fractions indicating how many frames each transform should take.
            E.g., [0.3, 0.2, 0.5] means:
            - first transform uses 30% of frames
            - second transform uses 20% of frames
            - third transform uses 50% of frames
            Must sum to 1.0. If None, splits time equally among transforms.
        """
        # Convert single transforms to lists for consistent handling
        self.transforms = [t if isinstance(t, (list, tuple)) else [t] for t in transforms]
        
        if frame_fractions is None:
            # Split frames equally
            self.frame_fractions = [1/len(self.transforms)] * len(self.transforms)
        else:
            if len(frame_fractions) != len(self.transforms):
                raise ValueError("Number of frame fractions must match number of transforms")
            if abs(sum(frame_fractions) - 1.0) > 1e-6:
                raise ValueError("Frame fractions must sum to 1.0")
            self.frame_fractions = frame_fractions

        # Convert fractions to cumulative ranges
        self.frame_splits = []
        current = 0
        for fraction in self.frame_fractions:
            self.frame_splits.append((current, current + fraction))
            current += fraction

    def __call__(self, surface: pv.PolyData, frame: int, total_frames: int) -> pv.PolyData:
        progress = frame / total_frames
        transformed_surface = surface.copy()
        combined_attrs = {}
        
        # Apply all transforms up to the current progress
        for (start, end), transform_list in zip(self.frame_splits, self.transforms):
            if progress <= start:
                # Haven't reached these transforms yet
                break
            elif progress >= end:
                # Apply the full transforms
                for transform in transform_list:
                    result = transform(transformed_surface, total_frames, total_frames)
                    if isinstance(result, tuple):
                        transformed_surface, transform_attr = result
                        combined_attrs.update(transform_attr)
                    else:
                        transformed_surface = result
            else:
                # Partially apply these transforms
                local_progress = (progress - start) / (end - start)
                local_frame = int(local_progress * total_frames)
                for transform in transform_list:
                    result = transform(transformed_surface, local_frame, total_frames)
                    if isinstance(result, tuple):
                        transformed_surface, transform_attr = result
                        combined_attrs.update(transform_attr)
                    else:
                        transformed_surface = result
                break
        
        return (transformed_surface, combined_attrs) if combined_attrs else transformed_surface

