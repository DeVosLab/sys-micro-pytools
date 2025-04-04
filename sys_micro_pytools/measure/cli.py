import click
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sys_micro_pytools.df import plate_grid2table
from sys_micro_pytools.visualize import get_nice_ticks
from sys_micro_pytools.measure import calibration_curve, calculate_ic_value

def empty_to_none(ctx, param, value):
    if value == ():
        return None
    return value

@click.group()
def measure():
    """Measure operations."""
    pass

@measure.command(name='dose-response')
@click.option('--filename_dose','-f', type=click.Path(exists=True), multiple=True, callback=empty_to_none,
              help='Path to dose response data')
@click.option('--grid2table', is_flag=True, default=False, 
              help='Convert dose response data from grid to table format')
@click.option('--var2add', type=click.STRING, default=None, 
              help='Variable to add to the dose response data')
@click.option('--var2add_value', type=click.STRING, multiple=True, default=None, 
              help='Value of the variable to add to the dose response data. This will be added as a new column to the dose response data with value for each row.')
@click.option('--filename_cal', type=click.Path(exists=True), default=None, 
              help='Path to calibration data')
@click.option('--ic_percentile', type=click.FLOAT, default=50, 
              help='Percentile of inhibition to calculate (e.g., 50 for IC50, 30 for IC30)')
@click.option('--threshold_pct', type=click.FLOAT, default=10, 
              help='Minimum percent change required to consider a valid dose-response')
@click.option('--groupby_vars', type=click.STRING, multiple=True, callback=empty_to_none, 
              help='Variables to group by, e.g. plate, rep')
@click.option('--condition_vars', type=click.STRING, multiple=True, callback=empty_to_none, 
              help='Variables of which unique combinations form a condition')
@click.option('--control_values', type=click.STRING, multiple=True, callback=empty_to_none, 
              help='Values of condition variables of the control condition')
@click.option('--control_same_var', type=click.STRING, default=None, 
              help='Variable of which the value is be the same for a subgroup of controls and a condition. This can be used to specify which controls are used for normalization, e.g. different DMSO dilutions for different compounds.')
@click.option('--dose_var', type=click.STRING, default='Dose', 
              help='Dose variable name')
@click.option('--response_var', type=click.STRING, default='Response', 
              help='Response variable name')
@click.option('--normalize_response_var', is_flag=True, default=False, 
              help='Normalize response variable')
@click.option('--inferred_var', type=click.STRING, default=None, 
              help='Variable to infer from response variable using the calibration curve')
@click.option('--log_scale_x', is_flag=True, default=False, 
              help='Log scale for x-axis')
@click.option('--log_scale_y', is_flag=True, default=False, 
              help='Log scale for y-axis')
@click.option('--ic_method', type=click.Choice(['relative', 'inner_range']), default='relative', 
              help='Whether to calculate the IC value relative to control, or within the range of the response variable for each treatment')
@click.option('--x_unit', type=click.STRING, default='µM', 
              help='Unit of the x-axis')
@click.option('--y_unit', type=click.STRING, default=None, 
              help='Unit of the y-axis')
@click.option('--row_var', type=click.STRING, default='row', 
              help='Row variable name for plotting layouts')
@click.option('--col_var', type=click.STRING, default='col', 
              help='Column variable name for plotting layouts')
def dose_response_cli(
    filename_dose, grid2table, var2add, var2add_value, filename_cal, 
    ic_percentile, threshold_pct, groupby_vars, condition_vars, control_values,
    control_same_var, dose_var, response_var, normalize_response_var, inferred_var,
    log_scale_x, log_scale_y, ic_method, x_unit, y_unit, row_var, col_var
):
    """Process dose response data."""

    # Get dose-response data
    filenames_dose = [Path(file) for file in filename_dose]
    dfs_dose = []
    for i, filename_dose in enumerate(filenames_dose):
        df_dose = pd.read_csv(filename_dose, header=0, index_col=False)
        if grid2table:
            df_dose = plate_grid2table(df_dose)
        if var2add is not None and var2add_value is not None:
            # Add the variable to the dose response data
            df_dose[var2add] = var2add_value[i]
        dfs_dose.append(df_dose)
    df_dose = pd.concat(dfs_dose)

    # Remove rows with NaN
    df_dose = df_dose[df_dose[response_var].notna()]

    # Convert to correct type
    if groupby_vars is not None:
        # Convert groupby variables to categorical
        for var in groupby_vars:
            df_dose[var] = df_dose[var].astype(pd.CategoricalDtype(categories=list(df_dose[var].unique()), ordered=True))
    else:
        # If no groupby variables are provided, create a dummy variable
        groupby_vars = ['groupby_dummy']
        df_dose[groupby_vars[0]] = 'All data'
    df_dose[row_var] = df_dose[row_var].astype(
        pd.CategoricalDtype(categories=list(df_dose[row_var].unique()), ordered=True)
        )
    df_dose[col_var] = df_dose[col_var].astype(
        pd.CategoricalDtype(categories=list(df_dose[col_var].unique()), ordered=True)
        )
    df_dose[dose_var] = df_dose[dose_var].astype(float)
    df_dose[response_var] = df_dose[response_var].astype(float)

    # Get the controls
    control_query = ' and '.join(
            f'{var} == "{value}"' if isinstance(value, str) else f'{var} == {value}'
            for var, value in zip(condition_vars, control_values)
        )
    df_control = df_dose.query(control_query)
    
    # Normalize response for each group to the control response
    if normalize_response_var:
        response_var_norm = f'{response_var}_norm'
        df_dose[response_var_norm] = np.nan
        if control_same_var is not None:
            normalizeby_vars = groupby_vars + [control_same_var]
        else:
            normalizeby_vars = groupby_vars
        for group_name, group_idx in df_dose.groupby(normalizeby_vars).groups.items():
            if not isinstance(group_name, tuple):
                group_name = (group_name,)
            query = ' and '.join(
                f'{var} == "{value}"' if isinstance(value, str) else f'{var} == {value}'
                for var, value in zip(normalizeby_vars, group_name)
            )
            df_control_group = df_control.query(query)
            df_dose.loc[group_idx, response_var_norm] = df_dose.loc[group_idx, response_var] / df_control_group[response_var].mean()
    
    # Get DMSO controls after normalization
    df_control = df_dose.query(control_query)

    # Get calibration data
    if inferred_var is not None and filename_cal is not None:
        # Set flag
        do_calibration = True
        x = response_var_norm if normalize_response_var else response_var
        y = inferred_var

        # Read the calibration data
        filename_cal = Path(filename_cal)
        df_cal = pd.read_csv(filename_cal, header=0, index_col=False)
        df_cal[response_var] = df_cal[response_var].astype(float)

        # Normalize the calibration data: mean of all controls
        if normalize_response_var:
            df_cal[response_var_norm] = df_cal[response_var] / df_control[response_var].mean()
        
        # Calculate the calibration curve
        model = calibration_curve(df_cal, x, y)
        slope = model.coef_[0][0]
        intercept = model.intercept_[0]
        r_squared = model.score( df_cal[[x]], df_cal[[y]])
        eq = f"y = {slope:.5f}x + {intercept:.5f} (R² = {r_squared:.2f})"
        
        # Calculate the standard error
        y_pred_train = model.predict(df_cal[[x]])
        residuals = df_cal[y].values - y_pred_train.flatten()
        
        # Mean Squared Error (MSE)
        mse = np.sum(residuals ** 2) / (len(df_cal) - 2)
        
        # Calculate x statistics
        x_mean = np.mean(df_cal[x])
        x_sum_sq_dev = np.sum((df_cal[x] - x_mean)**2)
        
        # Create prediction points for the plot
        x_range = np.linspace(df_cal[x].min(), df_cal[x].max(), 100)
        x_range_reshape = x_range.reshape(-1, 1)
        y_pred = model.predict(x_range_reshape).flatten()
        
        # t-value for 95% confidence
        t_value = 1.96
        
        # Calculate confidence interval (for mean response)
        confidence_intervals = t_value * np.sqrt(mse * (1/len(df_cal) + 
                                        ((x_range - x_mean)**2 / x_sum_sq_dev)))
        
        # Calculate prediction interval (for individual observations)
        # Note the addition of 1 inside the square root
        prediction_intervals = t_value * np.sqrt(mse * (1 + 1/len(df_cal) + 
                                        ((x_range - x_mean)**2 / x_sum_sq_dev)))

        # Plot the calibration curve
        plt.figure(figsize=(10, 8))
        sns.scatterplot(data=df_cal, x=x, y=y, color='black', label='Observed data')
        sns.lineplot(x=x_range, y=y_pred, color='red', label=eq)
        
        # Plot confidence interval (for mean)
        plt.fill_between(x_range, 
                       y_pred - confidence_intervals,
                       y_pred + confidence_intervals, 
                       color='#d62728', alpha=0.2, 
                       label='95% Confidence interval')
        
        # Plot prediction interval (for individual observations)
        plt.fill_between(x_range, 
                       y_pred - prediction_intervals,
                       y_pred + prediction_intervals, 
                       color='blue', alpha=0.1, 
                       label='95% Prediction interval')
                        
        plt.xlabel(response_var)
        plt.ylabel(inferred_var)
        plt.title(f"Calibration Curve (R-squared: {r_squared:.2f})")
        plt.legend()
        plt.grid(axis='both')
        plt.tight_layout()
    else:
        do_calibration = False
        x = response_var_norm if normalize_response_var else response_var
    
    if do_calibration:
        df_dose[y] = model.predict(df_dose[[x]])

    # Plot heatmap of (normalized) responses
    n_groups = df_dose.groupby(groupby_vars).ngroups
    n_rows = np.ceil(np.sqrt(n_groups)).astype(int)
    n_cols = np.ceil(n_groups / n_rows).astype(int)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 8))
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = [axes]
    vmin = df_dose[x].min()
    vmax = df_dose[x].max()
    for ax, (group_name, df_group) in zip(axes, df_dose.groupby(groupby_vars)):
        group_name = group_name
        df_p_pivot = df_group.pivot_table(
            values=x,
            index=row_var,
            columns=col_var,
            aggfunc='mean',
            observed=False,
            dropna=False)
        sns.heatmap(df_p_pivot, ax=ax, cmap='RdBu_r', vmin=vmin, vmax=vmax)
        ax.set_title(group_name)
    plt.suptitle(f'{x} heatmap')
    plt.tight_layout()

    # Get the compounds
    dose_query = ' or '.join(
        f'{var} != "{value}"' if isinstance(value, str) else f'{var} != {value}'
        for var, value in zip(condition_vars, control_values)
    )
    df_dose = df_dose.query(dose_query)
    condition_vars_no_dose = [var for var in condition_vars if var != dose_var]
    conditions = df_dose[condition_vars_no_dose].drop_duplicates()
    n_conditions = len(conditions)

    # Plot the dose-response curve for conditions
    x = dose_var
    y = response_var_norm if normalize_response_var else response_var
    n_rows = np.ceil(np.sqrt(n_conditions)).astype(int)
    n_cols = np.ceil(n_conditions / n_rows).astype(int)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(9, 9))
    axes = axes.flatten()
    x_margin = (df_dose[x].max() - df_dose[x].min()) * 0.1 if not log_scale_x else 0
    y_margin = (df_dose[y].max() - df_dose[y].min()) * 0.1 if not log_scale_y else 0
    for ax, (group_name, df_group) in zip(axes, df_dose.groupby(condition_vars_no_dose)):
        sns.boxplot(
            data=df_group, x=x, y=y, ax=ax, label=None, native_scale=True,
            log_scale=(log_scale_x, log_scale_y)
            )
        ax.set_xlim(df_dose[x].min() - x_margin, df_dose[x].max() + x_margin)
        ax.set_ylim(df_dose[y].min() - y_margin, df_dose[y].max() + y_margin)
        ax.set_xlabel(f'{x}')
        ax.set_ylabel(f'{y}')
        ax.set_title(group_name)
        ax.grid(True)
    plt.tight_layout()

    # Plot the inferred dose-response
    if do_calibration:
        x = dose_var
        y = inferred_var
        y_range = df_dose[y].max() - df_dose[y].min()
        y_margin = y_range * 0.1 if not log_scale_y else 0
        y_start, y_stop, y_step = get_nice_ticks(df_dose[y].min(), df_dose[y].max())
        x_margin = (df_dose[x].max() - df_dose[x].min()) * 0.1 if not log_scale_x else 0
        n_rows = np.ceil(np.sqrt(n_conditions)).astype(int)
        n_cols = np.ceil(n_conditions / n_rows).astype(int)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(9, 9))
        axes = axes.flatten()
        for ax, (group_name, df_group) in zip(axes, df_dose.groupby(condition_vars_no_dose)):
            sns.boxplot(
                data=df_group, x=x, y=y, ax=ax, label=None, native_scale=True, 
                log_scale=(log_scale_x, log_scale_y)
                )
            ax.set_xlim(df_group[x].min() - x_margin, df_group[x].max() + x_margin)
            ax.set_ylim(df_group[y].min() - y_margin, df_group[y].max() + y_margin)
            ax.set_xticks(df_group[x].unique())
            ax.set_xticklabels(df_group[x].unique(), rotation=45)
            ax.set_yticks(np.arange(y_start, y_stop, y_step))
            ax.set_yticklabels(np.arange(y_start, y_stop, y_step))
            ax.set_xlabel(f'{x}')
            ax.set_ylabel(f'{y}')
            ax.set_title(group_name)
            ax.grid(True)
        plt.tight_layout()
    
    plt.show()

    # Calculate and plot IC50 for each compound
    x = dose_var
    if do_calibration:
        y = inferred_var
    else:
        y = response_var_norm if normalize_response_var else response_var
    y_start, y_stop, y_step = get_nice_ticks(df_dose[y].min(), df_dose[y].max())
    ic_percentile = ic_percentile
    threshold_pct = threshold_pct
    control_value = df_control[y].mean()
    ic_value_results = {}
    
    n_rows = np.ceil(np.sqrt(n_conditions)).astype(int)
    n_cols = np.ceil(n_conditions / n_rows).astype(int)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 9))
    axes = axes.flatten()

    for ax, (group_name, df_group) in zip(axes, df_dose.groupby(condition_vars_no_dose)):
        # Group by dose and calculate mean and std
        df_grouped = df_group.groupby(x)[y].agg(['mean', 'std']).reset_index()
        
        # Calculate IC50 using control value as reference
        ic_value, params = calculate_ic_value(
            df_group,
            x_col=x,
            y_col=y, 
            control_value=control_value,
            ic_percentile=ic_percentile,
            threshold_pct=threshold_pct
            )
        ic_value_results[group_name] = {'ic_value': ic_value, 'params': params}
        
        # Plot the data and fitted curve
        ax.errorbar(df_grouped[x], df_grouped['mean'], yerr=df_grouped['std'], 
                fmt='o', capsize=5, label='Data')
        
        if ic_value is not None:
            # Generate points for the fitted curve
            x_fit = np.logspace(np.log10(df_grouped[x].min()*0.5), 
                            np.log10(df_grouped[x].max()*2), 100)
            y_fit = params['model'](x_fit, *params['params'])
            
            # Plot the fitted curve
            fit_label = f'IC{ic_percentile} = {ic_value:.2f} {x_unit} (R² = {params["r_squared"]:.2f})'
            ax.plot(x_fit, y_fit, 'r-', label=fit_label)
            
            # Mark the IC value point
            ic_y = params['model'](ic_value, *params['params'])
            ax.plot([ic_value], [ic_y], 'rx', markersize=10)
            ax.axvline(x=ic_value, color='r', linestyle='--', alpha=0.3)
            ax.axhline(y=ic_y, color='r', linestyle='--', alpha=0.3)
        else:
            # Add a note about why IC value couldn't be calculated
            status = params.get('status', 'unknown_error')
            ax.text(0.5, 0.5, f"No valid IC{ic_percentile}\n({status})", 
                transform=ax.transAxes, ha='center', fontsize=10,
                bbox=dict(facecolor='white', alpha=0.8))
        ax.set_xlim(df_group[x].min(), df_group[x].max())
        ax.set_ylim(df_group[y].min(), df_group[y].max())
        ax.set_yticks(np.arange(y_start, y_stop, y_step))
        ax.set_xscale('log' if log_scale_x else 'linear')
        ax.set_yscale('log' if log_scale_y else 'linear')
        ax.set_xlabel(f'{x} ({x_unit})')
        ax.set_ylabel(f'{y} ({y_unit if y_unit is not None else ""})')
        ax.set_title(group_name)
        ax.grid(True)
        ax.legend(loc='best', fontsize=8)
    
    plt.tight_layout()
    plt.show()
    
    # Print IC values for all compounds with quality indicators
    print(f"\nIC{ic_percentile} values:")
    for compound, result in sorted(ic_value_results.items()):
        ic_value = result['ic_value']
        params = result['params']
        if ic_value is not None:
            status = params.get('status', 'unknown')
            r_squared = params.get('r_squared', 0)
            print(f"{compound}: {ic_value:.2f} {x_unit} (R² = {r_squared:.2f}, status: {status})")
        else:
            status = params.get('status', 'unknown_error')
            print(f"{compound}: Could not calculate IC{ic_percentile} ({status})")


if __name__ == '__main__':
    measure()