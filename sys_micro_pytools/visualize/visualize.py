import pandas as pd
from typing import Union, Tuple, List
import itertools
from matplotlib import colormaps
from matplotlib import pyplot as plt


def create_palette(plate_layout: pd.DataFrame, hue_vars: Union[Tuple, List], cmap: str ='tab20', 
                   remove_hue_combo: Union[tuple, list, None] = None, 
                   hue_start_rgba: Union[tuple, list, None] = None) -> dict:
        # Get all values of hue_vars
        hue_var_vals = []
        for hue_var in hue_vars:
            if hue_var not in plate_layout.columns:
                raise ValueError(f'{hue_var} is not a column in the plate layout file')
            hue_var_vals.append(plate_layout[hue_var].drop_duplicates().values)

        # Make all combinations of all posible hue_var_vals pairs
        hue_order = list(itertools.product(*hue_var_vals))

        # Remove hue combinations
        if remove_hue_combo is not None:
            for hue_combo in remove_hue_combo:
                if hue_combo in hue_order:
                    hue_order.remove(hue_combo)
        
        # Create palette with remaining hue combinations
        try:
            cmap = colormaps[cmap]
        except KeyError:
            try:
                cmap = plt.cm.get_cmap(f'cet_{cmap}')
            except ValueError:
                raise ValueError(f'{cmap} is not a valid colormap')

        colors = [cmap(i) for i in range(len(hue_order))]
        if hue_start_rgba is not None:
            colors = (tuple(hue_start_rgba),) + tuple(colors[:-1])
        palette = {hue: color for hue, color in zip(hue_order, colors)}
        return palette, hue_order