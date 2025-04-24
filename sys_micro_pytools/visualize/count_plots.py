from typing import Union
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from sys_micro_pytools.io import read_tiff_or_nd2

def count_objects_in_mask(mask_file: Union[str, Path]) -> int:
    ''' Count the number of objects in a mask image
    
    Parameters
    ----------
    mask_file : str or Path
        Path to mask image
        
    Returns
    -------
    count : int
        Number of objects in the mask
    '''
    mask = read_tiff_or_nd2(str(mask_file), bundle_axes='yx').astype(int)
    
    # Count unique labels excluding background (0)
    unique_labels = np.unique(mask)
    count = len(unique_labels[unique_labels > 0])
    
    return count


def create_count_df(df: pd.DataFrame) -> pd.DataFrame:
    ''' Create dataframe with object counts for each image
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with image information
        
    Returns
    -------
    df_counts : pd.DataFrame
        Dataframe with object counts
    '''
    # Create a copy of the dataframe
    df_counts = df.copy()
    
    # Count objects in each mask
    counts = []
    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Counting objects"):
        filename = row['Filename']
        count = count_objects_in_mask(filename)
        counts.append(count)
    
    # Add counts to dataframe
    df_counts['Count'] = counts
    
    return df_counts


def create_count_plot(
    df: pd.DataFrame,
    condition_vars: Union[list, tuple, str],
    palette: dict,
    title: str = None,
    figsize: tuple = (8, 6),
    y_label: str = "Object Count",
    y_lim: tuple = None,
    box_width: float = 0.8,
    jitter: bool = True,
    plot_type: str = "box",
):
    ''' Create boxplot showing counts for each condition
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with count information
    condition_vars : list, tuple or str
        Variables to use for color encoding
    palette : dict
        Dictionary mapping condition values to colors
    title : str
        Title for the plot
    figsize : tuple
        Figure size (width, height)
    y_label : str
        Label for y-axis
    y_lim : tuple
        Limits for y-axis (min, max)
    box_width : float
        Width of the boxes
    jitter : bool
        Whether to add jitter to the points
    plot_type : str
        Type of plot to create ('box', 'violin', or 'swarm')
    
    Returns
    -------
    fig : matplotlib figure
        Figure with boxplot
    '''
    fig, ax = plt.subplots(figsize=figsize)
    
    # If condition_vars is a string, convert to list
    if isinstance(condition_vars, str):
        condition_vars = [condition_vars]
    
    # Create categorical column based on condition variables
    df['Condition'] = df.apply(
        lambda row: ', '.join([str(row[var]) for var in condition_vars]),
        axis=1
    )
    
    # Map condition names to colors
    condition_colors = {}
    for condition, color in palette.items():
        if isinstance(condition, tuple):
            condition_name = ', '.join([str(c) for c in condition])
        else:
            condition_name = str(condition)
        condition_colors[condition_name] = color
    
    # Sort conditions by first variable then second variable, etc.
    conditions = sorted(df['Condition'].unique())
    
    # Create boxplot or violin plot
    if plot_type == 'violin':
        sns.violinplot(
            x='Condition', 
            y='Count', 
            data=df, 
            hue='Condition',
            palette=condition_colors,
            width=box_width,
            ax=ax,
            order=conditions,
            legend=False
        )
    elif plot_type == 'swarm':
        sns.swarmplot(
            x='Condition', 
            y='Count', 
            data=df, 
            hue='Condition',
            palette=condition_colors,
            ax=ax,
            order=conditions,
            legend=False
        )
    else:  # Default to boxplot
        sns.boxplot(
            x='Condition', 
            y='Count', 
            data=df, 
            hue='Condition',
            palette=condition_colors,
            width=box_width,
            ax=ax,
            order=conditions,
            legend=False
        )
    
    # Add points with jitter
    if jitter and plot_type != 'swarm':
        sns.stripplot(
            x='Condition', 
            y='Count', 
            data=df, 
            color='black', 
            alpha=0.5, 
            jitter=True,
            ax=ax,
            order=conditions
        )
    
    # Set y-axis limits if provided
    if y_lim is not None:
        ax.set_ylim(y_lim)
    
    # Set labels and title
    ax.set_xlabel('')
    ax.set_ylabel(y_label, fontsize=14)
    if title:
        ax.set_title(title, fontsize=16)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.grid(True)
    plt.tight_layout()
    
    return fig