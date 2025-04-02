import numpy as np
from .visualize import create_palette

from .grid_plots import (
    create_grid_plot,    
    get_df_images
)

from .channel_plots import create_channel_plots

def get_nice_ticks(data_min, data_max):
    """
    Returns nice round numbers for tick start, stop, and step based on the data range.
    Uses 1, 2, 5, 10, 20, 50, 100, etc. as intervals.
    
    Parameters:
    -----------
    data_min : float
        Minimum value in the data
    data_max : float
        Maximum value in the data
        
    Returns:
    --------
    tuple : (start, stop, step)
        start: first tick mark
        stop: last tick mark
        step: spacing between ticks
    """
    data_range = data_max - data_min
    
    # Get the order of magnitude of the range
    magnitude = np.floor(np.log10(data_range))
    normalized_range = data_range / (10 ** magnitude)
    
    # Candidate intervals
    candidates = np.array([1, 2, 5, 10])
    # We want 4-10 tick marks, so divide range by these numbers
    n_ticks = normalized_range / candidates
    # Find the step that gives us a good number of ticks
    good_candidates = candidates[np.logical_and(n_ticks >= 4, n_ticks <= 10)]
    
    if len(good_candidates) > 0:
        step = good_candidates[0] * (10 ** magnitude)
    else:
        # If no good candidates, use the smallest one
        step = candidates[0] * (10 ** magnitude)
    
    # Calculate start and stop
    start = np.floor(data_min / step) * step
    stop = np.ceil(data_max / step) * step
    
    return start, stop, step

__all__ = [
    'create_palette',
    'create_channel_plots',
    'create_grid_plot',
    'get_df_images',
    'get_nice_ticks'
]
