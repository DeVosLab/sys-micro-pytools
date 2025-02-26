import pandas as pd
from typing import Union, Tuple, List
import itertools
from matplotlib import colormaps
from matplotlib import pyplot as plt


def create_palette(plate_layout: pd.DataFrame, condition_vars: Union[Tuple, List], 
                   cmap: str ='tab20', conditions2remove: Union[tuple, list, None] = None) -> dict:
    # Get all values of condition_vars
    condition_vals = []
    for condition_var in condition_vars:
        if condition_var not in plate_layout.columns:
            raise ValueError(f'{condition_var} is not a column in the plate layout file')
        condition_vals.append(plate_layout[condition_var].drop_duplicates().values)

    # Make all combinations of all posible condition_vals pairs
    condition_order = list(itertools.product(*condition_vals))

    # Remove condition combinations
    if conditions2remove is not None:
        for condition in conditions2remove:
            if condition in condition_order:
                condition_order.remove(condition)
    
    # Create palette with remaining condition combinations
    try:
        cmap = colormaps[cmap]
    except KeyError:
        try:
            cmap = plt.cm.get_cmap(f'cet_{cmap}')
        except ValueError:
            raise ValueError(f'{cmap} is not a valid colormap')
    colors = [cmap(i) for i in range(len(condition_order))]
    palette = {condition: color for condition, color in zip(condition_order, colors)}
    return palette, condition_order