from argparse import ArgumentParser
from typing import Union, Tuple, List, Literal
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
import re
import tifffile
from matplotlib import pyplot as plt
import seaborn as sns

from sys_micro_pytools.io import read_tiff_or_nd2
from sys_micro_pytools.df import link_df2plate_layout
from sys_micro_pytools.visualize import create_palette


def get_df_images(input_path: Union[str, Path], check_batches: bool, suffix: str,
                  filename_well_idx: Union[list, tuple], filename_field_idx: Union[list, tuple],
                  skip_wells: Union[list, tuple]) -> pd.DataFrame:
    ''' Get dataframe with image information

    Parameters
    ----------
    input_path : str or Path
        Path to directory containing images
    check_batches : bool
        Check if input_path contains subdirectories
    suffix : str
        Suffix of image files
    filename_well_idx : list or tuple
        Start and stop indices of the well name in the filename
    filename_field_idx : list or tuple
        Start and stop indices of the field number in the filename
    skip_wells : list or tuple
        List of wells to skip

    Returns
    -------
    df_images : pd.DataFrame
        Dataframe with image information
    '''
    assert isinstance(input_path, (str, Path)), 'input_path must be a string or Path object'
    assert isinstance(check_batches, bool), 'check_batches must be a boolean'
    assert isinstance(suffix, str), 'suffix must be a string'
    assert isinstance(filename_well_idx, (list, tuple)), 'filename_well_idx must be a list or tuple'
    assert isinstance(filename_field_idx, (list, tuple)), 'filename_field_idx must be a list or tuple'
    assert isinstance(skip_wells, (list, tuple)), 'skip_wells must be a list or tuple'

    input_path = Path(input_path)
    dirs = [d for d in input_path.iterdir() if d.is_dir()] if check_batches else [input_path]
    if dirs == []:
        dirs = [input_path]
    df_images = pd.DataFrame()
    for dir in dirs:
        # Use re.findall to extract numbers after "plate" and "R"
        plate_pattern = r'plate([A-Za-z0-9]+)'
        r_pattern = r'R(\d+)'
        plate_id = re.findall(plate_pattern, dir.stem)[0] if 'plate' in dir.stem else 1
        rep_id = re.findall(r_pattern, dir.stem)[0] if dir.stem[-2] == 'R' else 1

        files = sorted([f for f in dir.iterdir() if f.suffix == suffix and not f.name.startswith('.')])
        wells = [f.stem[filename_well_idx[0]:filename_well_idx[1]] for f in files]
        rows =  [w[0] for w in wells]
        cols = [int(w[1:]) for w in wells]
        fields = [f.stem[filename_field_idx[0]:filename_field_idx[1]] for f in files]

        df_entry = pd.DataFrame().from_dict({
            'Dir': [dir.stem] * len(files),
            'Rep': [int(rep_id)] * len(files),
            'Plate': [plate_id] * len(files),
            'Well': wells,
            'Row': rows,
            'Col': cols,
            'Field': fields,
            'Filename': files
        })

        df_images = pd.concat(
            [df_images if not df_images.empty else None, df_entry],
            axis=0,
            ignore_index=True
            ).reset_index(drop=True)

    # Remove wells to skip
    df_images = df_images[~df_images['Well'].isin(skip_wells)].reset_index(drop=True)

    return df_images


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


def main(**kwargs):
    input_path = Path(kwargs['input_path'])
    output_path = Path(kwargs['output_path'])
    output_path.mkdir(parents=True, exist_ok=True)

    if isinstance(kwargs['skip_wells'], str):
        kwargs['skip_wells'] = [kwargs['skip_wells']]

    if isinstance(kwargs['condition_vars'], str):
        kwargs['condition_vars'] = [kwargs['condition_vars']]
    
    if kwargs['conditions2remove'] is not None:
        kwargs['conditions2remove'] = [tuple(condition.split(',')) for condition in kwargs['conditions2remove']]

    # Set color palette for treatments
    plate_layout = pd.read_csv(kwargs['plate_layout'], sep=",|;", engine='python'); 
    plate_layout.columns = [col.capitalize() for col in plate_layout.columns]
    plate_layout['Plate'] = plate_layout['Plate'].astype(str)
    plate_layout['Well'] = plate_layout['Well'].astype(str)
    plate_layout['Row'] = plate_layout['Row'].astype(str)
    plate_layout['Col'] = plate_layout['Col'].astype(int)

    # Set color palette
    palette, _ = create_palette(
        plate_layout,
        condition_vars=kwargs['condition_vars'],
        cmap=kwargs['cmap'],
        conditions2remove=kwargs['conditions2remove'],
    )
    
    # Image path
    input_path = Path(kwargs['input_path'])
    df_images = get_df_images(
        input_path,
        kwargs['check_batches'],
        kwargs['suffix'],
        kwargs['filename_well_idx'],
        kwargs['filename_field_idx'],
        kwargs['skip_wells']
    )
    df_images.columns = [col.capitalize() for col in df_images.columns]
    df_images['Plate'] = df_images['Plate'].astype(str)
    df_images['Well'] = df_images['Well'].astype(str)
    df_images['Row'] = df_images['Row'].astype(str)
    df_images['Col'] = df_images['Col'].astype(int)
    df_images['Field'] = df_images['Field'].astype(int)

    # Merge plate layout with df_images
    df_images = link_df2plate_layout(df_images, plate_layout)
    
    # Count objects in masks
    df_counts = create_count_df(df_images)
    
    # For each directory, create a count boxplot
    for dir in tqdm(sorted(df_counts['Dir'].unique()), desc="Creating plots"):
        df_dir = df_counts[df_counts['Dir'] == dir]
        
        # Create boxplot
        fig = create_count_plot(
            df_dir,
            kwargs['condition_vars'],
            palette,
            title=f"{dir} - Object Counts",
            y_label=kwargs['y_label'],
            y_lim=kwargs['y_lim'],
            box_width=kwargs['box_width'],
            jitter=kwargs['jitter'],
            plot_type=kwargs['plot_type']
        )
        
        # Save figure
        filename = output_path.joinpath(f'{dir}_count_plot.{kwargs["output_format"]}')
        fig.savefig(
            filename,
            dpi=kwargs['dpi'],
            bbox_inches='tight'
            )
        plt.close(fig)
        print(f'Saved count plot to {filename}')
        
        # Save counts to CSV
        if kwargs['save_csv']:
            csv_filename = output_path.joinpath(f'{dir}_counts.csv')
            df_dir.to_csv(csv_filename, index=False)
            print(f'Saved counts to {csv_filename}')


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('-i', '--input_path', type=str, default=None,
        help='Path to directory containing mask images')
    parser.add_argument('-o', '--output_path', type=str, default=None,
        help='Path to directory where output will be saved')
    parser.add_argument('--plate_layout', type=str, default=None,
        help='Path to plate layout file')
    parser.add_argument('--suffix', type=str, default='.tif',
        help='Suffix of image files')
    parser.add_argument('--filename_well_idx', nargs=2, type=int, default=[4,7],
        help='Start and stop indices of the well name in the filename (default: [4,7])')
    parser.add_argument('--filename_field_idx', nargs=2, type=int, default=[17,21],
        help='Start and stop indices of the field number in the filename (default: [17,21])')
    parser.add_argument('--skip_wells', nargs='+', type=str, default=[],
        help='List of wells to skip')
    parser.add_argument('--cmap', type=str, default='cet_glasbey',
        help='Color map to use for boxplots')
    parser.add_argument('--condition_vars', nargs='*', type=str, default=['Treat', 'Dose'],
        help='Variables to use for color encoding')
    parser.add_argument('--conditions2remove', nargs='*', type=str, default=None,
        help='List of condition combinations to remove from the palette, e.g. --conditions2remove "DMSO,L"')
    parser.add_argument('--check_batches', action='store_true',
        help='Check if input_path contains subdirectories')
    parser.add_argument('--y_label', type=str, default='Object Count',
        help='Label for y-axis')
    parser.add_argument('--y_lim', nargs=2, type=float, default=None,
        help='Limits for y-axis (min, max)')
    parser.add_argument('--box_width', type=float, default=0.8,
        help='Width of the boxes in the boxplot')
    parser.add_argument('--jitter', action='store_true', default=True,
        help='Whether to add jitter to the points')
    parser.add_argument('--plot_type', type=str, default='box',
        choices=['box', 'violin', 'swarm'],
        help='Type of plot to create')
    parser.add_argument('--output_format', type=str, default='png',
        choices=['png', 'jpg', 'pdf', 'svg'],
        help='Format for output figures')
    parser.add_argument('--dpi', type=int, default=150,
        help='DPI for output figures')
    parser.add_argument('--save_csv', action='store_true', default=False,
        help='Whether to save counts to CSV file')
    args = parser.parse_args()
    
    return args


if __name__ == '__main__':
    args = parse_args()
    main(**vars(args))
