from argparse import ArgumentParser
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
import seaborn as sns
import re
from sys_micro_pytools.df import plate_grid2layout
from sys_micro_pytools.visualize import get_nice_ticks

def calibration_curve(df: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
    """
    Calculate the calibration curve for a given dataframe.
    """

    # Fit the calibration curve
    model = LinearRegression().fit(df[[x_col]], df[[y_col]])

    return model

def calculate_ic_value(df, x_col, y_col, ic_percentile=50, threshold_pct=10, control_value=None):
        """
        Calculate IC value using either a 4-parameter logistic regression model or a quadratic model,
        selecting the one with better fit.
        
        Parameters:
        -----------
        df : pandas DataFrame
            DataFrame containing dose-response data
        x_col : str
            Column name for dose values
        y_col : str
            Column name for response values
        control_value : float, optional
            Control value (no drug) to use as reference
        ic_percentile : float, optional
            Percentile of inhibition to calculate (e.g., 50 for IC50, 30 for IC30)
        threshold_pct : float, optional
            Minimum percent change required to consider a valid dose-response
            
        Returns:
        --------
        float
            IC value or None if no valid dose-response
        dict
            Parameters of the fitted model and quality metrics
        """
        from scipy.optimize import curve_fit
        import warnings
        
        def logistic_4pl(x, params):
            """4-parameter logistic function"""
            bottom, top, ic50, hill = params
            return bottom + (top - bottom) / (1 + (x / ic50) ** hill)
        
        def quadratic(x, params):
            """Quadratic function"""
            a, b, c = params
            return a * x**2 + b * x + c
        
        # Sort by x column
        df_sorted = df.sort_values(by=x_col)
        x_data = df_sorted[x_col].values
        y_data = df_sorted[y_col].values
        
        # Check if there's enough variation in the response
        y_min, y_max = min(y_data), max(y_data)
        y_range = y_max - y_min
        
        if control_value is not None:
            # Calculate percent inhibition relative to control
            max_inhibition_pct = 100 * abs(min(y_data) - control_value) / control_value
            if max_inhibition_pct < threshold_pct:
                return None, {
                    'status': 'insufficient_inhibition',
                    'max_inhibition_pct': max_inhibition_pct,
                    'threshold_pct': threshold_pct
                }
        elif y_range / y_max < threshold_pct/100:
            return None, {
                'status': 'flat_response',
                'y_range_pct': 100 * y_range / y_max,
                'threshold_pct': threshold_pct
            }
        
        # Try both fits
        fits = {}
        
        # 1. Try logistic fit
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                p0_logistic = [min(y_data), max(y_data), np.median(x_data), 1.0]
                bounds_logistic = (
                    [0, 0.5*max(y_data), min(x_data), -10],
                    [2*max(y_data), 2*max(y_data), 10*max(x_data), 10]
                )
                params_logistic, pcov_logistic = curve_fit(
                    logistic_4pl, x_data, y_data, 
                    p0=p0_logistic, bounds=bounds_logistic, maxfev=10000
                )
                
                y_pred_logistic = logistic_4pl(x_data, *params_logistic)
                ss_total = np.sum((y_data - np.mean(y_data))**2)
                ss_residual = np.sum((y_data - y_pred_logistic)**2)
                r_squared_logistic = 1 - (ss_residual / ss_total)
                
                fits['logistic'] = {
                    'params': params_logistic,
                    'pcov': pcov_logistic,
                    'r_squared': r_squared_logistic,
                    'model': logistic_4pl
                }
        except Exception:
            fits['logistic'] = None
            
        # 2. Try quadratic fit
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                p0_quad = [1, 1, np.mean(y_data)]
                params_quad, pcov_quad = curve_fit(quadratic, x_data, y_data, p0=p0_quad)
                
                y_pred_quad = quadratic(x_data, *params_quad)
                ss_total = np.sum((y_data - np.mean(y_data))**2)
                ss_residual = np.sum((y_data - y_pred_quad)**2)
                r_squared_quad = 1 - (ss_residual / ss_total)
                
                fits['quadratic'] = {
                    'params': params_quad,
                    'pcov': pcov_quad,
                    'r_squared': r_squared_quad,
                    'model': quadratic
                }
        except Exception:
            fits['quadratic'] = None
            
        # Select best fit
        best_fit = None
        best_r_squared = -np.inf
        best_model = None
        
        for model_name, fit in fits.items():
            if fit is not None and fit['r_squared'] > best_r_squared:
                best_fit = fit
                best_r_squared = fit['r_squared']
                best_model = model_name
                
        if best_fit is None:
            return None, {'status': 'all_fits_failed'}
            
        # Calculate IC value based on best fit
        if best_model == 'logistic':
            bottom, top, ic50, hill = best_fit['params']
            response_level = bottom + (top - bottom) * (100 - ic_percentile) / 100
            
            if hill != 0:
                ic_value = ic50 * ((top - bottom) / (response_level - bottom) - 1) ** (1 / hill)
            else:
                ic_value = None
                
        else:  # quadratic
            a, b, c = best_fit['params']
            target_y = np.mean(y_data) * (100 - ic_percentile) / 100
            # Solve quadratic equation: ax² + bx + (c - target_y) = 0
            discriminant = b**2 - 4*a*(c - target_y)
            
            if discriminant >= 0:
                x1 = (-b + np.sqrt(discriminant)) / (2*a)
                x2 = (-b - np.sqrt(discriminant)) / (2*a)
                # Choose the root that's within our data range
                if min(x_data) <= x1 <= max(x_data):
                    ic_value = x1
                elif min(x_data) <= x2 <= max(x_data):
                    ic_value = x2
                else:
                    ic_value = None
            else:
                ic_value = None
        
        # Check if IC value is within the tested x range
        if ic_value is None:
            status = 'calculation_failed'
        elif ic_value < 0.5*min(x_data) or ic_value > 2*max(x_data):
            status = 'extrapolated_ic_value'
        elif best_r_squared < 0.7:
            status = 'poor_fit'
        else:
            status = 'good_fit'
            
        return ic_value, {
            'model_type': best_model,
            'params': best_fit['params'],
            'r_squared': best_r_squared,
            'status': status,
            'std_errors': np.sqrt(np.diag(best_fit['pcov'])),
            'model': best_fit['model']
        }


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--filename_dose', type=Path, nargs='+', required=True,
                        help='Path(s) to dose response data')
    parser.add_argument('--grid2list', action='store_true', default=False,
                        help='Convert dose response data from grid to list format')
    parser.add_argument('--var2add', type=str, default=None,
                        help='Variable to add to the dose response data')
    parser.add_argument('--var2add_value', type=str, nargs='*', default=None,
                        help=('Value of the variable to add to the dose response data.'
                              'This will be added as a new column to the dose response data with value for each row.'))
    parser.add_argument('--filename_cal', type=Path, default=None,
                        help='Path to calibration data')
    parser.add_argument('--ic_percentile', type=float, default=50,
                        help='Percentile of inhibition to calculate (e.g., 50 for IC50, 30 for IC30)')
    parser.add_argument('--threshold_pct', type=float, default=10,
                        help='Minimum percent change required to consider a valid dose-response')
    parser.add_argument('--groupby_vars', type=str, nargs='+', default=None,
                        help='Variables to group by, e.g. plate, rep')
    parser.add_argument('--condition_vars', type=str, nargs='+', default=None,
                        help='Variables of which unique combinations form a condition')
    parser.add_argument('--control_values', type=float, nargs='+', default=None,
                        help='Values of condition variables of the control condition')
    parser.add_argument('--control_same_var', type=str, default=None,
                        help=('Variable of which the value is be the same for a subgroup of controls and a condition.'
                              'This can be used to specify which controls are used for normalization, e.g. different DMSO dilutions '
                              'for different compounds.'))
    parser.add_argument('--dose_var', type=str, default='Dose',
                        help='Dose variable name')
    parser.add_argument('--response_var', type=str, default='Response',
                        help='Response variable name')
    parser.add_argument('--normalize_response_var', action='store_true', default=False,
                        help='Normalize response variable')
    parser.add_argument('--inferred_var', type=str, default=None,
                        help='Variable to infer from response variable using the calibration curve')
    parser.add_argument('--log_scale_x', action='store_true', default=False,
                        help='Log scale for x-axis')
    parser.add_argument('--log_scale_y', action='store_true', default=False,
                        help='Log scale for y-axis')
    parser.add_argument('--ic_method', type=str, choices=['relative', 'inner_range'], default='relative',
                        help=('Whether to calculate the IC value relative to control, '
                              'or within the range of the response variable for each treatment'))
    parser.add_argument('--row_var', type=str, default='row',
                        help='Row variable name for plotting layouts')
    parser.add_argument('--col_var', type=str, default='col',
                        help='Column variable name for plotting layouts')
    
    if isinstance(args.filename_dose, str):
        args.filename_dose = [Path(args.filename_dose)]
    if isinstance(args.condition_vars, str):
        args.condition_vars = [args.condition_vars]
    if isinstance(args.control_values, str):
        args.control_values = [args.control_values]
    return args


def main(args):
    if args.inferred_var is not None and args.filename_cal is not None:
        # Set flag
        do_calibration = True

        # Read the calibration data
        filename_cal = Path(args.filename_cal)
        df_cal = pd.read_csv(filename_cal, header=0, index_col=False)
        df_cal[args.dose_var] = df_cal[args.dose_var].astype(float)
        df_cal[args.response_var] = df_cal[args.response_var].astype(float)
        
        # Calculate the calibration curve
        model = calibration_curve(df_cal, args.inferred_var, args.response_var)
        print(model.coef_)
        slope = model.coef_[0][0]
        intercept = model.intercept_[0]
        eq = f"y = {slope:.5f}x + {intercept:.5f}"
        r_squared = model.score(df_cal[[args.inferred_var]], df_cal[args.response_var])

        # Calculate the standard error
        y_pred_train = model.predict(df_cal[[args.inferred_var]])
        residuals = df_cal[args.response_var].values - y_pred_train.flatten()
        residual_std = np.sqrt(np.sum(residuals ** 2) / (len(df_cal) - 2))
        x_mean = np.mean(df_cal[args.inferred_var])
        x_std = np.sum((df_cal[args.inferred_var] - x_mean)**2)
        x_range = np.linspace(df_cal[args.inferred_var].min(), df_cal[args.inferred_var].max(), 100)
        y_pred = model.predict(x_range.reshape(-1, 1)).flatten()
        std_error = residual_std * np.sqrt(1 + 1/len(df_cal) + (x_range - x_mean)**2 / x_std)

        print(f"Standard error: {residual_std:.2f}")

        # Plot the calibration curve
        df_cal[args.inferred_var] = model.predict(df_cal[[args.response_var]])
        plt.figure(figsize=(10, 8))
        sns.scatterplot(data=df_cal, x=args.response_var, y=args.inferred_var, color='black', label='Observed data')
        sns.lineplot(x=df_cal[args.inferred_var], y=df_cal[args.response_var], color='red', label=eq)
        plt.fill_between(x_range, y_pred - 1.96*std_error, y_pred + 1.96*std_error, 
                        color='#d62728', alpha=0.2, label='95% Confidence interval')
        plt.xlabel(args.response_var)
        plt.ylabel(args.inferred_var)
        plt.title(f"Calibration Curve (R-squared: {r_squared:.2f})")
        plt.legend()
        plt.grid(axis='both')
        plt.tight_layout()
    else:
        do_calibration = False

    # Calculate the dose response curve
    filenames_dose = [Path(file) for file in args.filename_dose]
    dfs_dose = []
    for i, filename_dose in enumerate(filenames_dose):
        df_dose = pd.read_csv(filename_dose, header=0, index_col=False)
        if args.grid2list:
            df_dose = plate_grid2layout(df_dose)
        if args.var2add is not None and args.var2add_value is not None:
            # Add the variable to the dose response data
            df_dose[args.var2add] = args.var2add_value[i]
        dfs_dose.append(df_dose)
    df_dose = pd.concat(dfs_dose)

    # Remove rows with NaN
    df_dose = df_dose[df_dose[args.response_var].notna()]

    # Convert to correct type
    if args.groupby_vars is not None:
        # Convert groupby variables to categorical
        for var in args.groupby_vars:
            df_dose[var] = df_dose[var].astype(pd.CategoricalDtype(categories=list(df_dose[var].unique()), ordered=True))
    else:
        # If no groupby variables are provided, create a dummy variable
        args.groupby_vars = ['groupby_dummy']
        df_dose[args.groupby_vars[0]] = 'All data'
    df_dose[args.row_var] = df_dose[args.row_var].astype(
        pd.CategoricalDtype(categories=list(df_dose[args.row_var].unique()), ordered=True)
        )
    df_dose[args.col_var] = df_dose[args.col_var].astype(
        pd.CategoricalDtype(categories=list(df_dose[args.col_var].unique()), ordered=True)
        )
    df_dose[args.dose_var] = df_dose[args.dose_var].astype(float)
    df_dose[args.response_var] = df_dose[args.response_var].astype(float)
    if do_calibration:
        df_dose[args.inferred_var] = (df_dose[args.response_var] - intercept) / slope

    # Get DMSO controls
    control_query = ' and '.join(
            f'{var} == "{value}"' if isinstance(value, str) else f'{var} == {value}'
            for var, value in zip(args.condition_vars, args.control_values)
        )
    df_control = df_dose.query(control_query)
    n_controls = len(df_control)

    # Normalize response for each group to the control response
    if args.normalize_response_var:
        response_var_norm = f'{args.response_var}_norm'
        df_dose[response_var_norm] = np.nan
        if args.control_same_var is not None:
            normalizeby_vars = args.groupby_vars + [args.control_same_var]
        else:
            normalizeby_vars = args.groupby_vars
        for group_name, group_idx in df_dose.groupby(normalizeby_vars).groups.items():
            df_control_group = df_control.query(group_name)
            df_dose.loc[group_idx, response_var_norm] = df_dose.loc[group_idx, args.response_var] / df_control_group[args.response_var].mean()
    
    # Get DMSO controls after normalization
    df_control = df_dose.query(control_query)
    n_controls = len(df_control)

    # Plot heatmap of (normalized) responses
    y = response_var_norm if args.normalize_response_var else args.response_var
    n_groups = df_dose.groupby(args.groupby_vars).ngroups
    n_rows = np.ceil(np.sqrt(n_groups))
    n_cols = np.ceil(n_groups / n_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 8))
    axes = axes.flatten()
    vmin = df_dose[y].min()
    vmax = df_dose[y].max()
    for ax, (group_name, df_group) in zip(axes, df_dose.groupby(args.groupby_vars)):
        group_name = group_name
        df_p_pivot = df_group.pivot_table(
            values=y,
            index=args.row_var,
            columns=args.col_var,
            aggfunc='mean',
            observed=False,
            dropna=False)
        sns.heatmap(df_p_pivot, ax=ax, cmap='RdBu_r', vmin=vmin, vmax=vmax)
        ax.set_title(group_name)
    plt.suptitle(f'{y} heatmap')
    plt.tight_layout()

    # Plot dose response curves
    dilutions = df_control['DMSO_dilution'].unique()
    
    control_response_mean = {}
    control_response_std = {}
    control_response_ci = {}
    control_response_mean_ci = {}
    for dilution in dilutions:
        df_c = df_control[df_control['DMSO_dilution'] == dilution]
        control_response_mean[dilution] = df_c[y].mean()
        control_response_std[dilution] = df_c[y].std()
        control_response_ci[dilution] = 1.96 * control_response_std[dilution] / np.sqrt(n_controls)
        control_response_mean_ci[dilution] = (control_response_mean[dilution] - control_response_ci[dilution], control_response_mean[dilution] + control_response_ci[dilution])
        print(f"DMSO response mean ({dilution}): {control_response_mean[dilution]:.2f} (CI: {control_response_mean_ci[dilution][0]:.2f} - {control_response_mean_ci[dilution][1]:.2f})")
    
    ## Density
    control_density_mean = {}
    control_density_std = {}
    control_density_ci = {}
    control_density_mean_ci = {}
    for dilution in dilutions:
        df_c = df_control[df_control['DMSO_dilution'] == dilution]
        control_density_mean[dilution] = df_c['Density_pred'].mean()
        control_density_std[dilution] = df_c['Density_pred'].std()
        control_density_ci[dilution] = 1.96 * control_density_std[dilution] / np.sqrt(n_controls)
        control_density_mean_ci[dilution] = (control_density_mean[dilution] - control_density_ci[dilution], control_density_mean[dilution] + control_density_ci[dilution])
        print(f"DMSO density mean ({dilution}): {control_density_mean[dilution]:.2f} (CI: {control_density_mean_ci[dilution][0]:.2f} - {control_density_mean_ci[dilution][1]:.2f})")

    ## Counts
    control_counts_mean = {}
    control_counts_std = {}
    control_counts_ci = {}
    control_counts_mean_ci = {}
    for dilution in dilutions:
        df_c = df_control[df_control['DMSO_dilution'] == dilution]
        control_counts_mean[dilution] = df_c[y].mean()
        control_counts_std[dilution] = df_c[y].std()
        control_counts_ci[dilution] = 1.96 * control_counts_std[dilution] / np.sqrt(n_controls)
        control_counts_mean_ci[dilution] = (control_counts_mean[dilution] - control_counts_ci[dilution], control_counts_mean[dilution] + control_counts_ci[dilution])
        print(f"DMSO counts mean ({dilution}): {control_counts_mean[dilution]:.2f} (CI: {control_counts_mean_ci[dilution][0]:.2f} - {control_counts_mean_ci[dilution][1]:.2f})")

    # Get the compounds
    dose_query = ' or '.join(
        f'{var} != "{value}"' if isinstance(value, str) else f'{var} != {value}'
        for var, value in zip(args.condition_vars, args.control_values)
    )
    df_dose = df_dose.query(dose_query)
    condition_vars_no_dose = [var for var in args.condition_vars if var != x]
    conditions = df_dose[condition_vars_no_dose].drop_duplicates()
    n_conditions = len(conditions)

    # Plot the dose response curve for different compounds
    x = args.dose_var
    y = args.response_var_norm if args.normalize_response_var else args.response_var
    n_rows = np.ceil(np.sqrt(n_conditions)).astype(int)
    n_cols = np.ceil(n_conditions / n_rows).astype(int)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(9, 9))
    axes = axes.flatten()
    x_margin = (df_dose[x].max() - df_dose[x].min()) * 0.1
    y_margin = (df_dose[y].max() - df_dose[y].min()) * 0.1
    for ax, (group_name, df_group) in zip(axes, df_dose.groupby(condition_vars_no_dose)):
        sns.boxplot(
            data=df_group, x=x, y=y, ax=ax, label=None, native_scale=True,
            log_scale=(args.log_scale_x, args.log_scale_y)
            )
        ax.set_xlim(df_dose[x].min() - x_margin, df_dose[x].max() + x_margin)
        ax.set_ylim(df_dose[y].min() - y_margin, df_dose[y].max() + y_margin)
        ax.set_xlabel(f'{x}')
        ax.set_ylabel(f'{y}')
        ax.set_title(group_name)
        ax.grid(True)
    plt.tight_layout()

    # Plot the predicted density
    if args.do_calibration:
        x = args.dose_var
        y = args.inferred_var
        y_range = df_dose[y].max() - df_dose[y].min()
        y_margin = y_range * 0.1
        y_start, y_stop, y_step = get_nice_ticks(df_dose[y].min(), df_dose[y].max())
        x_margin = (df_dose[x].max() - df_dose[x].min()) * 0.1
        n_rows = np.ceil(np.sqrt(n_conditions)).astype(int)
        n_cols = np.ceil(n_conditions / n_rows).astype(int)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(9, 9))
        axes = axes.flatten()
        for ax, (group_name, df_group) in zip(axes, df_dose.groupby(condition_vars_no_dose)):
            sns.boxplot(
                data=df_group, x=x, y=y, ax=ax, label=None, native_scale=True, 
                log_scale=(args.log_scale_x, args.log_scale_y)
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
    x = args.dose_var
    y = args.response_var_norm if args.normalize_response_var else args.response_var
    ic_percentile = 30
    threshold_pct = 10
    control_value = df_control[y].mean()
    ic_value_results = {}
    
    n_rows = np.ceil(np.sqrt(n_conditions)).astype(int)
    n_cols = np.ceil(n_conditions / n_rows).astype(int)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 12))
    axes = axes.flatten()

    for ax, (group_name, df_group) in zip(axes, df_dose.groupby(condition_vars_no_dose)):
        # Group by dose and calculate mean and std
        df_grouped = df_group.groupby(x)[y].agg(['mean', 'std']).reset_index()
        
        # Calculate IC50 using control value as reference
        ic_value, params = calculate_ic_value(
            df_grouped,
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
            fit_label = f'IC{ic_percentile} = {ic_value:.2f} nM (R² = {params["r_squared"]:.2f})'
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
        ax.set_xlim(df_grouped[x].min(), df_grouped[x].max())
        ax.set_ylim(df_grouped[y].min(), df_grouped[y].max())
        ax.set_yticks(np.arange(df_grouped[y].min(), df_grouped[y].max(), y_step[y]))
        ax.set_xscale('log' if args.log_scale_x else 'linear')
        ax.set_yscale('log' if args.log_scale_y else 'linear')
        ax.set_xlabel(x)
        ax.set_ylabel(y)
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
            print(f"{compound}: {ic_value:.2f} nM (R² = {r_squared:.2f}, status: {status})")
        else:
            status = params.get('status', 'unknown_error')
            print(f"{compound}: Could not calculate IC{ic_value} ({status})")

if __name__ == '__main__':
    args = parse_args()
    main(args)

    