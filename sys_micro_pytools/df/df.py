import pandas as pd
from typing import Union


def link_df2plate_layout(df: pd.DataFrame, plate_layout: pd.DataFrame, on: Union[list, tuple] = ['Plate', 'Well']) -> pd.DataFrame:
    ''' Link image information to plate layout

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with image information
    plate_layout : pd.DataFrame
        Dataframe or series with plate layout information
    
    Returns
    -------
    df : pd.DataFrame
        Dataframe with original and plate layout information

    '''
    assert isinstance(df, pd.DataFrame), 'df_images must be a pandas DataFrame'
    assert isinstance(plate_layout, pd.DataFrame), 'plate_layout must be a pandas DataFrame'

    # Original columns
    orig_cols = list(df.columns)

    # Merge plate layout with df_images
    df = df.merge(plate_layout, how='left', on=on).reset_index(drop=True)

    # Remove column 'index' from plate_layout
    if 'index' in df.columns:
        df.drop(columns=['index'], inplace=True)

    # Remove duplicate cols by looping over original columns 
    # and use the column from plate_layout if it exists
    for col in orig_cols:
        if (col not in on) and (col in plate_layout.columns):
            df[col] = df[f'{col}_y']
            df.drop(columns=[f'{col}_x', f'{col}_y'], inplace=True)
    
    return df