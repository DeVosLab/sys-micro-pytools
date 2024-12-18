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


filename = "/Users/timvdl/Downloads/Composite.tif"
img = tifffile.imread(filename).astype(float)
img = np.moveaxis(img, 0, 1)
C, Z, Y, X = img.shape
print(img.shape)

# Normalize channels
pmin, pmax = 0.1, 99.9
for c in range(C):
    pmin_val, pmax_val = np.percentile(img[c,], (pmin, pmax))
    img[c,] = (img[c,] - pmin_val) / (pmax_val - pmin_val)


cyto_channel = 0
mito_channel = 1
nuclei_channel = 2
nuclei_img = img[nuclei_channel, :, :, :]
mito_img = img[mito_channel, :, :, :]
cyto_img = img[cyto_channel, :, :, :]

# Nuclei segmentation
nuclei_img = median_filter(nuclei_img, size=3)
nuclei_img_bw = nuclei_img > threshold_otsu(nuclei_img)
nuclei_img_bw = remove_small_objects(nuclei_img_bw, min_size=500)
nuclei_img_bw = clear_border(nuclei_img_bw)
nuclei_img_bw = binary_closing(nuclei_img_bw, footprint=np.ones((3, 3, 3)))
nuclei_img_bw = binary_opening(nuclei_img_bw, footprint=np.ones((2, 2, 2)))
nuclei_img_bw = binary_fill_holes(nuclei_img_bw)
nuclei_masks = label(nuclei_img_bw)

# Mitochondria segmentation
mito_img = median_filter(mito_img, size=3)
mito_img_bw = mito_img > threshold_otsu(mito_img)
mito_img_bw = remove_small_objects(mito_img_bw, min_size=10)
mito_img_bw = clear_border(mito_img_bw)
mito_img_bw = binary_closing(mito_img_bw, footprint=np.ones((1, 1, 1)))
mito_img_bw = binary_opening(mito_img_bw, footprint=np.ones((1, 1, 1)))
mito_img_bw = binary_fill_holes(mito_img_bw)
mito_masks = label(mito_img_bw)

# Cytoplasm segmentation
cyto_img = median_filter(cyto_img, size=3)

# Remove background above and below monolayer
cyto_masks = tifffile.imread("/Users/timvdl/Downloads/cyto_masks.tif")
cyto_masks = cyto_masks == 2
cyto_masks = binary_dilation(cyto_masks, footprint=ball(3))

nuclei_masks = np.where(cyto_masks, nuclei_masks, 0)
nuclei_masks = binary_erosion(nuclei_masks, footprint=ball(1)) > 0
nuclei_masks = remove_small_objects(nuclei_masks, min_size=1000)
mito_masks = np.where(cyto_masks, mito_masks, 0) > 0


masks = np.zeros_like(cyto_masks, dtype=int)
masks = np.where(cyto_masks, 3, masks)
masks = np.where(nuclei_masks, 1, masks)
masks = np.where(mito_masks, 2, masks)


print(np.unique(masks))

surfaces = labels2surface3D(masks, scale=(0.29, 0.26, 0.26))
print(f"Surfaces: {surfaces.keys()}")

render_settings = RenderSettings(
    background_color=(0.9, 0.9, 0.9),
    lighting="three lights",
    show_axes=True,
    axes_line_width=5,
    axes_labels=False
)

surface_attributes = {
    1: SurfaceAttributes(color="cyan", opacity=0.9, pbr=True, metallic=0.8, diffuse=1, specular=0.5),
    2: SurfaceAttributes(color="yellow", opacity=0.9, pbr=True, metallic=0.8, diffuse=1, specular=0.5),
    3: SurfaceAttributes(color="magenta", opacity=0.5, pbr=True, metallic=0.8, diffuse=1, specular=0.5)
}

#render_surfaces(surfaces, render_settings, surface_attributes, opacity=0.9)

def animate_surfaces(
    surfaces: Dict[int, pv.PolyData],
    transformations: Dict[int, List[Callable]],
    n_frames: int = 100,
    render_settings: Optional[RenderSettings] = None,
    surface_attributes: Optional[Dict[int, SurfaceAttributes]] = None,
    opacity: Optional[float] = None,
    output_path: str = "animation.gif",
    camera_position: str = "3d"
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
                for transform_func in transformations[label]:
                    current_surface = transform_func(current_surface, frame, n_frames)

                # Remove old actor and add new one
                plotter.remove_actor(actors[label])
                if surface_attributes and label in surface_attributes:
                    attrs = surface_attributes[label]
                else:
                    attrs = default_attributes
                    attrs.color = color_mapping[label]
                
                if opacity is not None:
                    attrs.opacity = opacity
                
                actors[label] = plotter.add_mesh(
                    current_surface,
                    color=attrs.color,
                    opacity=attrs.opacity,
                    pbr=attrs.pbr,
                    metallic=attrs.metallic,
                    diffuse=attrs.diffuse,
                    specular=attrs.specular
                )
        
        # Write the frame
        plotter.write_frame()

    # Close and finalize the animation
    plotter.close()

# Example transformation functions
def rotate(axis: Union[np.ndarray, List[float]], angle_degrees: float = 360, center: Union[np.ndarray, List[float]]=None):
    axis = np.array(axis) / np.linalg.norm(axis)
    def transform(surface: pv.PolyData, frame: int, total_frames: int, center: Union[np.ndarray, List[float]]=center) -> pv.PolyData:
        # Get the center of the object
        if center is None:
            center = surface.center
        else:
            center = np.array(center)
        
        # Create a copy of the surface
        transformed = surface.copy()
        
        # Move to origin
        translation = [-c for c in center]
        transformed.translate(translation, inplace=True)
        
        # Rotate
        progress = frame / total_frames
        current_angle = angle_degrees * progress
        transformed.rotate_vector(axis, current_angle, inplace=True)
        
        # Move back to original position
        transformed.translate(center, inplace=True)
        
        return transformed
    return transform

def translate(direction: Union[np.ndarray, List[float]], distance: float = 10):
    direction = np.array(direction) / np.linalg.norm(direction)
    def transform(surface: pv.PolyData, frame: int, total_frames: int) -> pv.PolyData:
        progress = frame / total_frames
        current_distance = distance * progress
        return surface.translate(direction * current_distance, inplace=True)
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
        surface.prop.opacity = 0
        return surface
    return transform

@dataclass
class Transform:
    func: Callable
    start_frame: int
    end_frame: int

class SequentialTransform:
    def __init__(self, *transforms: Callable, frame_splits: Optional[List[float]] = None):
        """
        Parameters
        ----------
        transforms : Callable
            Transform functions to apply sequentially
        frame_splits : Optional[List[float]]
            List of fractions indicating when each transform should start.
            E.g., [0.5, 0.8] means first transform runs 0-0.5, second 0.5-0.8, third 0.8-1.0
            If None, splits time equally among transforms
        """
        self.transforms = transforms
        if frame_splits is None:
            # Split frames equally
            splits = np.linspace(0, 1, len(transforms) + 1)
            self.frame_splits = list(zip(splits[:-1], splits[1:]))
        else:
            # Convert user splits to ranges
            splits = [0] + frame_splits + [1.0]
            self.frame_splits = list(zip(splits[:-1], splits[1:]))

    def __call__(self, surface: pv.PolyData, frame: int, total_frames: int) -> pv.PolyData:
        progress = frame / total_frames
        transformed_surface = surface.copy()
        
        # Apply all transforms up to the current progress
        for (start, end), transform in zip(self.frame_splits, self.transforms):
            if progress <= start:
                # Haven't reached this transform yet
                break
            elif progress >= end:
                # Apply the full transform
                transformed_surface = transform(transformed_surface, total_frames, total_frames)
            else:
                # Partially apply this transform
                local_progress = (progress - start) / (end - start)
                local_frame = int(local_progress * total_frames)
                transformed_surface = transform(transformed_surface, local_frame, total_frames)
                break
        
        return transformed_surface


# Add red XY plane that translates along Z to the surfaces
plane_center = surfaces[1].center
surfaces.update({
    4: pv.Plane(center=(plane_center[0], plane_center[1], 0), direction=(0, 0, 1), i_size=50, j_size=50)
})

surface_attributes.update({
    4: SurfaceAttributes(color="red", opacity=1)
})

# Move other surfaces to left
for label in [1, 2, 3]:
    surfaces[label] = surfaces[label].translate((0, -50, 0), inplace=True)

# Transformations
distance = 50
degrees = 2*360
transformations = {
    1: [
        SequentialTransform(
            translate([0, 1, 0], distance),
            wait(),
            translate([0, 1, 0], distance),
            frame_splits=[0.2, 0.5]
        )
    ],
    2: [
        SequentialTransform(
            translate([0, 1, 0], distance),
            wait(),
            translate([0, 1, 0], distance),
            frame_splits=[0.2, 0.5]
        )
    ],
    3: [
        SequentialTransform(
            translate([0, 1, 0], distance),
            wait(),
            translate([0, 1, 0], distance),
            frame_splits=[0.2, 0.5]
        )
    ],
    4: [
        SequentialTransform(
            wait(),
            translate([0, 0, 1], 50),
            wait(),
            frame_splits=[0.2, 0.5]
        )
    ]
}

animate_surfaces(
    surfaces,
    transformations,
    n_frames=64,
    render_settings=render_settings,
    surface_attributes=surface_attributes,
    output_path="cell_animation.gif",
    camera_position="yz"
)
