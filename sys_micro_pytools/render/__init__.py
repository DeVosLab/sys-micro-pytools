"""3D rendering utilities using PyVista."""

from .render import (
    RenderSettings,
    SurfaceAttributes,
    render_surfaces,
    labels2surface3D,
    animate_surfaces,
    generate_distinct_colors
)

from .volume import (
    VolumeSettings,
    create_multichannel_volume,
    create_isosurface_multichannel,
    render_volume_with_slices
)

__all__ = [
    'RenderSettings',
    'SurfaceAttributes', 
    'render_surfaces',
    'labels2surface3D',
    'animate_surfaces',
    'generate_distinct_colors',
    'VolumeSettings',
    'create_multichannel_volume',
    'create_isosurface_multichannel',
    'render_volume_with_slices'
]
