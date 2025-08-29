"""
MEASURING DOSE RESPONSE WINDOW: defines the window for the DoseResponse measuring tool
o   this script contains all UI elements for the tool and
o   accepts a callback to return to the main menu
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QLabel, QLineEdit,
    QCheckBox, QComboBox, QFileDialog, QMessageBox)

from sys_micro_pytools.df.plate_grid2table import plate_grid2table
from sys_micro_pytools.measure import calibration_curve, calculate_ic_value
from sys_micro_pytools.visualize import get_nice_ticks

class MeasureDoseResponseWindow(QWidget):
    """Window for measuring dose response."""

    def __init__(self, back):
        super().__init__()
        self.setWindowTitle("Measure Dose Response")
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Options to Measure Dose Response"))

        ## DOSE RESPONSE MEASUREMENT OPTIONS BELOW ##

        # select path to file containing dose response data
        self.dose_response_file_btn = QPushButton('Select file(s) containing dose response data')
        self.dose_response_file_btn.clicked.connect(self.select_dose_response_file)
        self.dose_response_file_label = QLabel('No file selected')
        layout.addWidget(self.dose_response_file_btn)
        layout.addWidget(self.dose_response_file_label)

        # convert dose response data from grid to table format
        self.grid2table_cb = QCheckBox('Convert dose response data to table format')
        layout.addWidget(self.grid2table_cb)

        # variable to add to dose response data
        self.var2add_line = QLineEdit()
        layout.addWidget(QLabel('Add variable to dose response data? (optional)'))
        layout.addWidget(self.var2add_line)

        # value of variable to add as new column to dose response data
        self.var2add_val_line = QLineEdit()
        layout.addWidget(QLabel('Value(s) of variable to add to dose response data (comma-separated):'))
        layout.addWidget(self.var2add_val_line)

        # select path to file containing calibration data
        self.calibration_data_file_btn = QPushButton('Select file containing calibration data (optional)')
        self.calibration_data_file_btn.clicked.connect(self.select_calibration_data)
        self.calibration_data_file_label = QLabel('No file selected')
        layout.addWidget(self.calibration_data_file_btn)
        layout.addWidget(self.calibration_data_file_label)

        # percentile of inhibition to calculate
        self.ic_percentile_line = QLineEdit('50')
        layout.addWidget(QLabel('Percentile of inhibition to calculate (e.g., 50 for IC50, 30 for IC30):'))
        layout.addWidget(self.ic_percentile_line)

        # min. % change required to consider valid dose response
        self.threshold_pct_line = QLineEdit('10')
        layout.addWidget(QLabel('Minimum percent change required to consider a valid dose-response:'))
        layout.addWidget(self.threshold_pct_line)

        # variables to group by
        self.groupby_vars_line = QLineEdit()
        layout.addWidget(QLabel('Variables to group by (comma-separated, optional):'))
        layout.addWidget(self.groupby_vars_line)

        # variables of which unique combinations form a condition
        self.condition_vars_line = QLineEdit()
        layout.addWidget(QLabel('Variables of which unique combinations form a condition (comma-separated, optional):'))
        layout.addWidget(self.condition_vars_line)

        # control condition values
        self.control_val_line = QLineEdit()
        layout.addWidget(QLabel('Values of condition variables of the control condition (comma-separated, optional):'))
        layout.addWidget(self.control_val_line)

        # variable where the value is same for subgroup of controls and a condition
        self.control_same_var_line = QLineEdit()
        layout.addWidget(QLabel('Variable of which the value is to be the same for a subgroup of conrols and a condition (optional):'))
        layout.addWidget(self.control_same_var_line)

        # dose variable name
        self.dose_var_line = QLineEdit('Dose')
        layout.addWidget(QLabel('Dose variable name:'))
        layout.addWidget(self.dose_var_line)

        # response variable name
        self.response_var_line = QLineEdit('Response')
        layout.addWidget(QLabel('Response variable name:'))
        layout.addWidget(self.response_var_line)

        # normalise response variable
        self.normalise_response_var_cb = QCheckBox('Normalise response variable')
        layout.addWidget(self.normalise_response_var_cb)

        # variable to infer from response variable using calibration# curve
        self.inferred_var_line = QLineEdit()
        layout.addWidget(QLabel('Variable to infer from response variable using the calibration curve (optional):'))
        layout.addWidget(self.inferred_var_line)

        # log scale for x/y-axes
        self.log_scale_x_cb = QCheckBox('Log scale for x-axis')
        self.log_scale_y_cb = QCheckBox('Log scale for y-axis')
        layout.addWidget(self.log_scale_x_cb)
        layout.addWidget(self.log_scale_y_cb)

        # calculate IC value relative to control, or within range of response variable for each treatment
        self.ic_method_combo = QComboBox()
        self.ic_method_combo.addItems(['relative', 'inner range'])
        layout.addWidget(QLabel('Method of calculating the IC value; whether relative to control, or within the range of the response variable for each treatment'))
        layout.addWidget(self.ic_method_combo)

        # unit of the x-axis
        self.x_unit_line = QLineEdit('µM')
        layout.addWidget(QLabel('Unit of the x-axis:'))
        layout.addWidget(self.x_unit_line)

        # unit of the y-axis
        self.y_unit_line = QLineEdit()
        layout.addWidget(QLabel('Unit of the y-axis (optional):'))
        layout.addWidget(self.y_unit_line)

        # row/column variable names for plotting layouts
        self.row_var_line = QLineEdit('row')
        self.col_var_line = QLineEdit('col')
        layout.addWidget(QLabel('Row variable name for plotting layouts:'))
        layout.addWidget(self.row_var_line)
        layout.addWidget(QLabel('Column variable name for plotting layouts:'))
        layout.addWidget(self.col_var_line)     

        # run or done button to begin processing
        self.run_btn = QPushButton('Measure Dose Response')
        self.run_btn.clicked.connect(self.measure_dose_response)
        layout.addWidget(self.run_btn)

        # back button to return to previous page
        self.back_btn = QPushButton("Back")
        self.back_btn.clicked.connect(back)
        layout.addWidget(self.back_btn)

        self.setLayout(layout)

        self.dose_files = []
        self.cal_file = None

    ## LOADING IN FILES & FOLDERS ##

    # specifies to load in the files
    def select_dose_response_file(self):
        dose_file, _ = QFileDialog.getOpenFileName(self, 'Select file(s) containing dose response data')
        if dose_file:
            self.dose_files = dose_file
            self.dose_response_file_label.setText(', '.join([Path(f).name for f in dose_file]))

    def select_calibration_data(self):
        cal_file, _ = QFileDialog.getOpenFileName(self, 'Select file containing calibration data')
        if cal_file:
            self.cal_file = cal_file
            self.calibration_data_file_label.setText(Path(cal_file).name)

    ## GATHERING AND PROCESSING INPUT ##

    # gathering all inputs for measuring dose response
    def measure_dose_response(self):

        if not self.dose_files:
            QMessageBox.warning(self, 'Missing Input', 'Please select at least one file containing the dose response data.')
        else:
            dose_file = self.dose_files
        dose_var = self.dose_var_line.text()
        response_var = self.response_var_line.text()
        x_unit = self.x_unit_line.text()
        row_var = self.row_var_line.text()
        col_var = self.col_var_line.text()
        grid2table = self.grid2table_cb.isChecked()
        normalise = self.normalise_response_var_cb.isChecked()
        log_x = self.log_scale_x_cb.isChecked()
        log_y = self.log_scale_y_cb.isChecked()
        ic_method = self.ic_method_combo.currentText()

        # optional variables
        var2add = self.var2add_line.text() if self.var2add_line.text() else None
        cal_file = self.cal_file or None
        control_same_var = self.control_same_var_line.text() if self.control_same_var_line.text() else None
        inferred_var = self.inferred_var_line.text() if self.inferred_var_line.text() else None
        y_unit = self.y_unit_line.text() if self.y_unit_line.text() else None

        # convert strings to integers
        var2add_val = self.var2add_val_line.text().split(',') if self.var2add_val_line.text() else None
        ic_percentile = float(self.ic_percentile_line.text())
        threshold_pct = float(self.threshold_pct_line.text())

        # optional variables, only split at ',' if text exists
        groupby_vars = self.groupby_vars_line.text().split(',') if self.groupby_vars_line.text() else None
        condition_vars = self.condition_vars_line.text().split(',') if self.condition_vars_line.text() else None
        control_val = self.control_val_line.text().split(',') if self.control_val_line.text() else None

        try:             
            # get dose-response data
            dose_files = [Path(file) for file in dose_file]
            dfs_dose = []
            for i, dose_file in enumerate(dose_files):
                df_dose = pd.read_csv(dose_file, header=0, index_col=False)
                if grid2table:
                    df_dose = plate_grid2table(df_dose)
                if var2add is not None and var2add_val is not None:
                    # Add the variable to the dose response data
                    df_dose[var2add] = var2add_val[i]
                dfs_dose.append(df_dose)
            df_dose = pd.concat(dfs_dose)

            # remove rows with NaN
            df_dose = df_dose[df_dose[response_var].notna()]

            # convert to correct type
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
                    for var, value in zip(condition_vars, control_val)
                )
            df_control = df_dose.query(control_query)
            
            # Normalize response for each group to the control response
            if normalise:
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
            if inferred_var is not None and cal_file is not None:
                # Set flag
                do_calibration = True
                x = response_var_norm if normalise else response_var
                y = inferred_var

                # Read the calibration data
                cal_file = Path(cal_file)
                df_cal = pd.read_csv(cal_file, header=0, index_col=False)
                df_cal[response_var] = df_cal[response_var].astype(float)

                # Normalize the calibration data: mean of all controls
                if normalise:
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
                x = response_var_norm if normalise else response_var
            
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
                for var, value in zip(condition_vars, control_val)
            )
            df_dose = df_dose.query(dose_query)
            condition_vars_no_dose = [var for var in condition_vars if var != dose_var]
            conditions = df_dose[condition_vars_no_dose].drop_duplicates()
            n_conditions = len(conditions)

            # Plot the dose-response curve for conditions
            x = dose_var
            y = response_var_norm if normalise else response_var
            n_rows = np.ceil(np.sqrt(n_conditions)).astype(int)
            n_cols = np.ceil(n_conditions / n_rows).astype(int)
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(9, 9))
            axes = axes.flatten()
            x_margin = (df_dose[x].max() - df_dose[x].min()) * 0.1 if not log_x else 0
            y_margin = (df_dose[y].max() - df_dose[y].min()) * 0.1 if not log_y else 0
            for ax, (group_name, df_group) in zip(axes, df_dose.groupby(condition_vars_no_dose)):
                sns.boxplot(
                    data=df_group, x=x, y=y, ax=ax, label=None, native_scale=True,
                    log_scale=(log_x, log_y)
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
                y_margin = y_range * 0.1 if not log_y else 0
                y_start, y_stop, y_step = get_nice_ticks(df_dose[y].min(), df_dose[y].max())
                x_margin = (df_dose[x].max() - df_dose[x].min()) * 0.1 if not log_x else 0
                n_rows = np.ceil(np.sqrt(n_conditions)).astype(int)
                n_cols = np.ceil(n_conditions / n_rows).astype(int)
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(9, 9))
                axes = axes.flatten()
                for ax, (group_name, df_group) in zip(axes, df_dose.groupby(condition_vars_no_dose)):
                    sns.boxplot(
                        data=df_group, x=x, y=y, ax=ax, label=None, native_scale=True, 
                        log_scale=(log_x, log_y)
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
                y = response_var_norm if normalise else response_var
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
                ax.set_xscale('log' if log_x else 'linear')
                ax.set_yscale('log' if log_y else 'linear')
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

            QMessageBox.information(self, 'Success', 'Dose response analysis completed!')
        except Exception as e:
            QMessageBox.critical(self, 'Error', str(e))