import sys
from pathlib import Path
import tifffile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QLineEdit, QFileDialog, QCheckBox,
    QSpinBox, QMessageBox, QComboBox)

from sys_micro_pytools.df import link_df2plate_layout
from sys_micro_pytools.df.plate_grid2table import plate_grid2table, plot_layout
from sys_micro_pytools.measure import calibration_curve, calculate_ic_value
from sys_micro_pytools.preprocess.flat_field import get_flat_field_files
from sys_micro_pytools.visualize import get_nice_ticks, create_palette
from sys_micro_pytools.visualize.channel_plots import create_channel_plots
from sys_micro_pytools.visualize.grid_plots import get_df_images, create_grid_plot
from sys_micro_pytools.visualize.count_plots import create_count_df, create_count_plot

"""
=============================================================
                      MAIN / HOME PAGE
=============================================================
three buttons that will redirect the user to different tools:
1)  Grid2Table
2)  Measure Dose Response
3)  Visualise Plots
=============================================================
"""
class MainMenu(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SysMicroPyTools GUI")
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Select a tool:"))

        self.grid2table_btn = QPushButton("Grid2Table")
        self.measure_dose_response_btn = QPushButton("Measure Dose Response")
        self.visualise_btn = QPushButton("Visualise Plots")

        layout.addWidget(self.grid2table_btn)
        layout.addWidget(self.measure_dose_response_btn)
        layout.addWidget(self.visualise_btn)

        self.setLayout(layout)

"""
=============================================================
                      GRID 2 TABLE TOOL
=============================================================
several parameters/options to customise programme settings:
- select input file
- select output folder
* filename of output file
* visualise plate layout
* index of plate to be visualised
- row names
- column names
* order of variables in visualisation
- number of columns for subplots in visualisation
* add annotations to heatmap
* treat variables as numeric in visualisation
* remove rows containing NA values from dataframe
                                                    optional*
=============================================================
"""
class Grid2TableWindow(QWidget):
    def __init__(self, back_callback):
        super().__init__()
        self.setWindowTitle("Grid2Table")
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Grid2Table Options"))

        # select path to input file
        self.input_path_btn = QPushButton('*Select input file')
        self.input_path_btn.clicked.connect(self.select_input_file)
        self.input_path_label = QLabel('No file selected')
        layout.addWidget(self.input_path_btn)
        layout.addWidget(self.input_path_label)

        # select path to save the merged layout dataframe
        # if not specified, output will be saved in same directory as input file(s)
        self.output_path_btn = QPushButton('Select output folder')
        self.output_path_btn.clicked.connect(self.select_output_folder)
        self.output_path_label = QLabel('No folder selected')
        layout.addWidget(self.output_path_btn)
        layout.addWidget(self.output_path_label)
        
        # write name of output file
        self.filename_line = QLineEdit()
        layout.addWidget(QLabel('Output filename (.csv):'))
        layout.addWidget(self.filename_line)

        # check box to visualise plate layout
        self.visualize_cb = QCheckBox('Visualise plate layout')
        layout.addWidget(self.visualize_cb)

        # index of plate to be visualised
        # if not specified, all plates will be visualised
        self.plate_id_line = QLineEdit()
        layout.addWidget(QLabel('Plate ID (comma-separated, optional):'))
        layout.addWidget(self.plate_id_line)

        # categories for row variable, i.e. row names
        self.row_categories_line = QLineEdit('A,B,C,D,E,F,G,H')
        layout.addWidget(QLabel('*Row categories (comma-separated):'))
        layout.addWidget(self.row_categories_line)

        # categories for column variable, i.e. column names
        self.col_categories_line = QLineEdit('1,2,3,4,5,6,7,8,9,10,11,12')
        layout.addWidget(QLabel('*Column categories (comma-separated):'))
        layout.addWidget(self.col_categories_line)

        # order of variables in the visualisation
        self.var_order_line = QLineEdit()
        layout.addWidget(QLabel('Order of variables (comma-separated, optional):'))
        layout.addWidget(self.var_order_line)

        # number of columns for the subplots in the visualisation
        self.ncols_spin = QSpinBox()
        self.ncols_spin.setMinimum(1)
        self.ncols_spin.setValue(3)
        layout.addWidget(QLabel('*Number of columns for subplots:'))
        layout.addWidget(self.ncols_spin)

        # check box to add annotations to the heatmap
        self.add_annot_cb = QCheckBox('Add annotations to heatmap')
        layout.addWidget(self.add_annot_cb)

        # variables to be treated as numeric in the visualisation
        self.numeric_vars_line = QLineEdit()
        layout.addWidget(QLabel('Numeric variables (comma-separated, optional):'))
        layout.addWidget(self.numeric_vars_line)

        # check box to remove rows containing NA values from the dataframe
        self.remove_na_cb = QCheckBox('Remove rows with NA values')
        layout.addWidget(self.remove_na_cb)

        layout.addWidget(QLabel('Required fields.*'))

        # run or done button to begin processing
        self.run_btn = QPushButton('Run Grid2Table')
        self.run_btn.clicked.connect(self.run_plate_grid2table)
        layout.addWidget(self.run_btn)

        self.back_btn = QPushButton("Back")
        self.back_btn.clicked.connect(back_callback)
        layout.addWidget(self.back_btn)
        self.setLayout(layout)

    # specifies to load in a file
    def select_input_file(self):
        input_file, _ = QFileDialog.getOpenFileName(self, 'Select input file')
        if input_file:
            self.input_path_label.setText(input_file)

    # specifies to load in a folder
    def select_output_folder(self):
        output_folder = QFileDialog.getExistingDirectory(self, 'Select output folder')
        if output_folder:
            self.output_path_label.setText(output_folder)

    # gathering all inputs for grid2table
    def run_plate_grid2table(self):

        input_path = self.input_path_label.text()
        ncols = self.ncols_spin.value()
        visualize = self.visualize_cb.isChecked()
        add_annot = self.add_annot_cb.isChecked()
        remove_rows_with_na = self.remove_na_cb.isChecked()

        # optional variables
        output_path = self.output_path_label.text() or None
        filename = self.filename_line.text() or None
        plate_id = self.plate_id_line.text() or None

        # split at ','
        row_categories = self.row_categories_line.text().split(',')
        # convert strings to integers
        col_categories = [int(x) for x in self.col_categories_line.text().split(',')]

        # optional variables, only split at ',' if text exists
        var_order = self.var_order_line.text().split(',') if self.var_order_line.text() else None
        numeric_vars = self.numeric_vars_line.text().split(',') if self.numeric_vars_line.text() else None

        try:
            df = plate_grid2table(input_path, remove_rows_with_na)

            if output_path:
                from pathlib import Path
                # if no filename, use input path as output filename
                filename_path = Path(filename) if filename else Path(input_path).with_suffix('.csv')
                full_path = Path(output_path) / filename_path
                df.to_csv(full_path, index=False)

            if visualize:
                if plate_id:
                    plate_ids = [x.strip() for x in plate_id.split(',')]
                else:
                    plate_ids = sorted(df['plate'].unique())
                for plate in plate_ids:
                    plot_layout(
                        df, plate, row_categories, col_categories, var_order, ncols, add_annot, numeric_vars
                    )

            QMessageBox.information(self, 'Success', 'Operation completed successfully!')

        except Exception as e:
            QMessageBox.critical(self, 'Error', str(e))

"""
=============================================================
                MEASURING DOSE RESPONSE TOOL
=============================================================
several parameters/options to customise programme settings:
- select dose response data file
* convert dose response data from grid to table format
* variable to add to the dose response data
* value of variable to be added ^ added as new column
* select file containing calibration data
- percentile of inhibition to calculate
- min. % change required to consider valid dose response
- variables to group by
- variables of which unique combinations form a condition
- control condition values
* variable where the value is same for subgroup of controls
  and a condition
- dose variable name
- response variable name
* normalise response variable
* variable to infer from response variable using calibration
  curve
* log scale for x-axis
* log scale for y-axis
- calculate IC value relative to control, or within range of
  response variable for each treatment
- unit of the x-axis
* unit of the y-axis
- row variable name for plotting layouts
- volumn variable name for plotting layouts
                                                    optional*
=============================================================
"""
class MeasureDoseResponseWindow(QWidget):
    def __init__(self, back_callback):
        super().__init__()
        self.setWindowTitle("Measure Dose Response")
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Options to Measure Dose Response"))

        # select path to file containing dose response data
        self.dose_response_file_btn = QPushButton('*Select file(s) containing dose response data')
        self.dose_response_file_btn.clicked.connect(self.select_dose_response_file)
        self.dose_response_file_label = QLabel('No file selected')
        layout.addWidget(self.dose_response_file_btn)
        layout.addWidget(self.dose_response_file_label)

        # convert dose response data from grid to table format
        self.grid2table_cb = QCheckBox('Convert dose response data to table format')
        layout.addWidget(self.grid2table_cb)

        # variable to add to dose response data
        self.var2add_line = QLineEdit()
        layout.addWidget(QLabel('Add variable to dose response data (optional):'))
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
        layout.addWidget(QLabel('*Percentile of inhibition to calculate (e.g., 50 for IC50, 30 for IC30):'))
        layout.addWidget(self.ic_percentile_line)

        # min. % change required to consider valid dose response
        self.threshold_pct_line = QLineEdit('10')
        layout.addWidget(QLabel('*Minimum percent change required to consider a valid dose-response:'))
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
        layout.addWidget(QLabel('*Dose variable name:'))
        layout.addWidget(self.dose_var_line)

        # response variable name
        self.response_var_line = QLineEdit('Response')
        layout.addWidget(QLabel('*Response variable name:'))
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
        layout.addWidget(QLabel('*Method of calculating the IC value; whether relative to control, or within the range of the response variable for each treatment'))
        layout.addWidget(self.ic_method_combo)

        # unit of the x-axis
        self.x_unit_line = QLineEdit('µM')
        layout.addWidget(QLabel('*Unit of the x-axis:'))
        layout.addWidget(self.x_unit_line)

        # unit of the y-axis
        self.y_unit_line = QLineEdit()
        layout.addWidget(QLabel('Unit of the y-axis (optional):'))
        layout.addWidget(self.y_unit_line)

        # row/column variable names for plotting layouts
        self.row_var_line = QLineEdit('row')
        self.col_var_line = QLineEdit('col')
        layout.addWidget(QLabel('*Row variable name for plotting layouts:'))
        layout.addWidget(self.row_var_line)
        layout.addWidget(QLabel('*Column variable name for plotting layouts:'))
        layout.addWidget(self.col_var_line)     

        layout.addWidget(QLabel('Required fields.*'))

        # run or done button to begin processing
        self.run_btn = QPushButton('Measure Dose Response')
        self.run_btn.clicked.connect(self.measure_dose_response)
        layout.addWidget(self.run_btn)

        self.back_btn = QPushButton("Back")
        self.back_btn.clicked.connect(back_callback)
        layout.addWidget(self.back_btn)
        self.setLayout(layout)

        self.dose_files = []
        self.cal_file = None

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

    # gathering all inputs for measuring dose response
    def measure_dose_response(self):

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

"""
=============================================================
                    VISUALISE PLOTS TOOL
=============================================================
three selections to visualise plots:
1)  count plot
2)  grid plot
3)  channel plot
=============================================================
"""
class VisualiseWindow(QWidget):
    def __init__(self, back_callback):
        super().__init__()
        self.setWindowTitle("Visualise Plots")
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Select an operation:"))

        self.countplot_btn = QPushButton('Count Plot')
        self.gridplot_btn = QPushButton('Grid Plot')
        self.channelplot_btn = QPushButton('Channel Plot')

        layout.addWidget(self.countplot_btn)
        layout.addWidget(self.gridplot_btn)
        layout.addWidget(self.channelplot_btn)

        self.back_btn = QPushButton("Back")
        self.back_btn.clicked.connect(back_callback)
        layout.addWidget(self.back_btn)
        self.setLayout(layout)

"""
=============================================================
                        COUNT PLOT
=============================================================
several parameters/options to customise programme settings:
- select input file
- select output folder
- select file containing plate layout
- suffix of image files
- start and stop indices of well name in filename
- start and stop indices of field number in filename
* list of wells to skip
- colour map to use for boxplots
- variables to use for colour encoding
* list of condition combinations to remove from the palette
* check if input path contains subdirectories
- label for y-axis
* limits for y-axis
- width of boxes in the boxplot
* add jitter to the points
- type of plot to create
- format for output figures
- DPI for output figures
* save counts to CSV file
* plate ID to use; overwrites plate_layout automatically
* rep ID to use; overwrites plate_layout automatically
                                                    optional*
=============================================================
"""
class CountPlotWindow(QWidget):
    def __init__(self, back_callback):
        super().__init__()
        self.setWindowTitle("Count Plot")
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Count Plot Options"))

        # select path to input file
        self.input_path_btn = QPushButton('*Select input file')
        self.input_path_btn.clicked.connect(self.select_input_file)
        self.input_path_label = QLabel('No file selected')
        layout.addWidget(self.input_path_btn)
        layout.addWidget(self.input_path_label)        

        # select path to output folder
        self.output_path_btn = QPushButton('*Select output folder')
        self.output_path_btn.clicked.connect(self.select_output_folder)
        self.output_path_label = QLabel('No folder selected')
        layout.addWidget(self.output_path_btn)
        layout.addWidget(self.output_path_label)

        # select path to file containing plate layout
        self.plate_layout_btn = QPushButton('*Select file containing plate layout')
        self.plate_layout_btn.clicked.connect(self.select_layout_file)
        self.plate_layout_label = QLabel('No file selected')
        layout.addWidget(self.plate_layout_btn)
        layout.addWidget(self.plate_layout_label)

        # suffix of image files
        self.suffix_line = QLineEdit('.tif')
        layout.addWidget(QLabel('*Suffix of image files:'))
        layout.addWidget(self.suffix_line)

        # start and stop indices of well name in filename
        self.well_idx_line = QLineEdit('4,7')
        layout.addWidget(QLabel('*Indices of well name in filename (start,stop):'))
        layout.addWidget(self.well_idx_line)

        # start and stop indices of field number in filename
        self.field_idx_line = QLineEdit('17,21')
        layout.addWidget(QLabel('*Indices of field number in filename (start,stop):'))
        layout.addWidget(self.field_idx_line)

        # list of wells to skip
        self.skip_wells_line = QLineEdit()
        layout.addWidget(QLabel('List of wells to skip (comma-separated, optional):'))
        layout.addWidget(self.skip_wells_line)

        # colour map to use for boxplots
        self.cmap_line = QLineEdit('cet_glasbey')
        layout.addWidget(QLabel('*Colour map to use for boxplots:'))
        layout.addWidget(self.cmap_line)

        # variables to use for colour encoding
        self.condition_vars_line = QLineEdit('Treat,Dose')
        layout.addWidget(QLabel('*Variables to use for colour encoding (comma-separated):'))
        layout.addWidget(self.condition_vars_line)

        # list of condition combinations to remove from the palette
        self.conditions2remove_line = QLineEdit()
        layout.addWidget(QLabel('List of condition combinations to remove from the palette (comma-separated, optional):'))
        layout.addWidget(self.conditions2remove_line)

        # check if input path contains subdirectories
        self.check_batches_cb = QCheckBox('Check if input path contains subdirectories')
        layout.addWidget(self.check_batches_cb)

        # label for y-axis
        self.y_label_line = QLineEdit('*Object Count')
        layout.addWidget(QLabel('Label for y-axis:'))
        layout.addWidget(self.y_label_line)

        # limits for y-axis
        self.y_lim_line = QLineEdit()
        layout.addWidget(QLabel('Limits for y-axis (min,max; optional):'))
        layout.addWidget(self.y_lim_line)

        # width of boxes in the boxplot
        self.box_width_line = QLineEdit('0.8')
        layout.addWidget(QLabel('*Width of boxes in the boxplot:'))
        layout.addWidget(self.box_width_line)

        # add jitter to the points
        self.jitter_cb = QCheckBox('Add jitter to the points')
        layout.addWidget(self.jitter_cb)

        # type of plot to create
        self.plot_type_combo = QComboBox()
        self.plot_type_combo.addItems(['box', 'violin', 'swarm'])
        layout.addWidget(QLabel('*Type of plot to create:'))
        layout.addWidget(self.plot_type_combo)

        # format for output figures
        self.output_format_line = QLineEdit('png')
        layout.addWidget(QLabel('*Format for output figures:'))
        layout.addWidget(self.output_format_line)

        # DPI for output figures
        self.dpi_line = QLineEdit('150')
        layout.addWidget(QLabel('*DPI for output figures:'))
        layout.addWidget(self.dpi_line)

        # save counts to CSV file
        self.save_csv_cb = QCheckBox('Save counts to CSV file')
        layout.addWidget(self.save_csv_cb)

        # plate ID to use; overwrites plate_layout automatically
        self.plate_id_line = QLineEdit()
        layout.addWidget(QLabel('Plate ID to use.\nOverwrites plate_layout automatically.\nUseful if plate ID is not in the filename. (optional):'))
        layout.addWidget(self.plate_id_line)

        # rep ID to use; overwrites plate_layout automatically
        self.rep_id_line = QLineEdit()
        layout.addWidget(QLabel('Rep ID to use.\nOverwrites plate_layout automatically.\nUseful if rep ID is not in the filename. (optional):'))
        layout.addWidget(self.rep_id_line)

        layout.addWidget(QLabel('Required fields.*'))

        # run or done button to begin processing
        self.run_btn = QPushButton('Visualise Count Plot')
        self.run_btn.clicked.connect(self.visualise_count_plot)
        layout.addWidget(self.run_btn)

        self.back_btn = QPushButton("Back")
        self.back_btn.clicked.connect(back_callback)
        layout.addWidget(self.back_btn)
        self.setLayout(layout)

    # specifies to load in the files and folder
    def select_input_file(self):
        input_file, _ = QFileDialog.getOpenFileName(self, 'Select input file')
        if input_file:
            self.input_path_label.setText(input_file)

    def select_output_folder(self):
        output_folder, _ = QFileDialog.getExistingDirectory(self, 'Select output folder')
        if output_folder:
            self.output_path_label.setText(output_folder)

    def select_layout_file(self):
        layout_file, _ = QFileDialog.getOpenFileName(self, 'Select file containing plate layout')
        if layout_file:
            self.plate_layout_label.setText(layout_file)

    # gathering all inputs for visualising count plot
    def visualise_count_plot(self):

        input_path = self.input_path_label.text()
        output_path = self.output_path_label.text()
        plate_layout = self.plate_layout_label.text()
        suffix = self.suffix_line.text()
        cmap = self.cmap_line.text()
        y_label = self.y_label_line.text()
        output_format = self.output_format_line.text()
        check_batches = self.check_batches_cb.isChecked()
        jitter = self.jitter_cb.isChecked()
        save_csv = self.save_csv_cb.isChecked()
        plot_type = self.plot_type_combo.currentText()

        # split at ','
        condition_vars = self.condition_vars_line.text().split(',')

        # optional variables
        plate_id = self.plate_id_line.text() if self.plate_id.text() else None
        rep_id = self.rep_id_line.text() if self.rep_id.text() else None

        # only split at ',' if text exists
        skip_wells = self.skip_wells_line.text().split(',') if self.skip_wells_line.text() else None
        conditions2remove = self.conditions2remove_line.text().split(',') if self.conditions2remove_line.text() else None

        # convert strings to integers
        box_width = float(self.box_width_line.text())
        dpi = int(self.dpi_line.text())
        well_idx = int(self.well_idx_line.text().split(','))
        field_idx = int(self.field_idx_line.text().split(','))
        y_lim = float(self.y_lim_line.text().split(',')) if self.y_lim_line.text() else None

        try:                
            input_path = Path(input_path)
            output_path = Path(output_path)
            output_path.mkdir(parents=True, exist_ok=True)

            if isinstance(skip_wells, str):
                skip_wells = [skip_wells]

            if isinstance(condition_vars, str):
                condition_vars = [condition_vars]
            
            if conditions2remove is not None:
                conditions2remove = [tuple(condition.split(',')) for condition in conditions2remove]

            # set color palette for treatments
            plate_layout = pd.read_csv(plate_layout, sep=",|;", engine='python'); 
            plate_layout.columns = [col.capitalize() for col in plate_layout.columns]
            plate_layout['Plate'] = plate_layout['Plate'].astype(str)
            plate_layout['Well'] = plate_layout['Well'].astype(str)
            plate_layout['Row'] = plate_layout['Row'].astype(str)
            plate_layout['Col'] = plate_layout['Col'].astype(int)

            # set color palette
            palette, _ = create_palette(
                plate_layout,
                condition_vars=condition_vars,
                cmap=cmap,
                conditions2remove=conditions2remove,
            )
            
            # image path
            input_path = Path(input_path)
            df_images = get_df_images(
                input_path,
                check_batches,
                suffix,
                well_idx,
                field_idx,
                skip_wells,
                plate_id,
                rep_id
            )
            df_images.columns = [col.capitalize() for col in df_images.columns]
            df_images['Plate'] = df_images['Plate'].astype(str)
            df_images['Well'] = df_images['Well'].astype(str)
            df_images['Row'] = df_images['Row'].astype(str)
            df_images['Col'] = df_images['Col'].astype(int)
            df_images['Field'] = df_images['Field'].astype(int)

            # merge plate layout with df_images
            df_images = link_df2plate_layout(df_images, plate_layout)
            
            # count objects in masks
            df_counts = create_count_df(df_images)
            
            # for each directory, create a count boxplot
            for dir in tqdm(sorted(df_counts['Dir'].unique()), desc="Creating plots"):
                df_dir = df_counts[df_counts['Dir'] == dir]
                
                # create boxplot
                fig = create_count_plot(
                    df_dir,
                    condition_vars,
                    palette,
                    title=f"{dir} - Object Counts",
                    y_label=y_label,
                    y_lim=y_lim,
                    box_width=box_width,
                    jitter=jitter,
                    plot_type=plot_type
                )
                
                # save figure
                filename = output_path.joinpath(f'{dir}_count_plot.{output_format}')
                fig.savefig(
                    filename,
                    dpi=dpi,
                    bbox_inches='tight'
                    )
                plt.close(fig)
                print(f'Saved count plot to {filename}')
                
                # save counts to CSV
                if save_csv:
                    csv_filename = output_path.joinpath(f'{dir}_counts.csv')
                    df_dir.to_csv(csv_filename, index=False)
                    print(f'Saved counts to {csv_filename}')

            QMessageBox.information(self, 'Success', 'Operation completed successfully!')
        
        except Exception as e:
            QMessageBox.critical(self, 'Error', str(e))

"""
=============================================================
                         GRID PLOT
=============================================================
several parameters/options to customise programme settings:
- select input file
- select output folder
- select file containing plate layout
- suffix of image files
- start and stop indices of well name in the filename
- start and stop indices of field number in the filename
* list of wells to skip
- type of image to plot
- channels to use for making the plots
* well(s) to use as reference for normalisation using its
  percentiles
* indicate that image is a mask. flat field correction will
  not be applied
* select file to image to use for flat field correction
- colour map to apply to image borders in grid plot
- variables to use for colour encoding
* list of condition combinations to remove from the palette
* check if input path contains subdirectories
* index of field to plot for each well. if none, random field
  will be selected
* plate ID to use; overwrites plate_layout automatically
* rep ID to use; overwrites plate_layout automatically
                                                    optional*
=============================================================
"""
class GridPlotWindow(QWidget):
    def __init__(self, back_callback):
        super().__init__()
        self.setWindowTitle("Grid Plot")
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Grid Plot Options"))

        # select path to input folder
        self.input_path_btn = QPushButton('*Select input folder')
        self.input_path_btn.clicked.connect(self.select_input_folder)
        self.input_path_label = QLabel('No folder selected')
        layout.addWidget(self.input_path_btn)
        layout.addWidget(self.input_path_label)

        # select path to output folder
        self.output_path_btn = QPushButton('*Select output folder')
        self.output_path_btn.clicked.connect(self.select_output_folder)
        self.output_path_label = QLabel('No folder selected')
        layout.addWidget(self.output_path_btn)
        layout.addWidget(self.output_path_label)

        # select path to file containing plate layout
        self.plate_layout_btn = QPushButton('*Select file containing plate layout')
        self.plate_layout_btn.clicked.connect(self.select_plate_layout)
        self.plate_layout_label = QLabel('No file selected')
        layout.addWidget(self.plate_layout_btn)
        layout.addWidget(self.plate_layout_label)

        # suffix of image files
        self.suffix_line = QLineEdit('.nd2')
        layout.addWidget(QLabel('*Suffix of image files:'))
        layout.addWidget(self.suffix_line)

        # start and stop indices of well name in the filename
        self.well_idx_line = QLineEdit('4,7')
        layout.addWidget(QLabel('*Indices of well name in filename (start,stop):'))
        layout.addWidget(self.well_idx_line)

        # start and stop indices of field number in the filename
        self.field_idx_line = QLineEdit('17,21')
        layout.addWidget(QLabel('*Indices of field number in filename (start,stop):'))
        layout.addWidget(self.field_idx_line)

        # list of wells to skip
        self.skip_wells_line = QLineEdit()
        layout.addWidget(QLabel('List of wells to skip (comma-separated, optional):'))
        layout.addWidget(self.skip_wells_line)

        # type of image to plot
        self.img_type_combo = QComboBox()
        self.img_type_combo.addItems(['grayscale', 'multichannel', 'mask'])
        layout.addWidget(QLabel('Type of image to plot'))
        layout.addWidget(self.img_type_combo)

        # channels to use for making the plots
        self.channels2use_line = QLineEdit('0')
        layout.addWidget(QLabel('*Channels to use for making the plots (comma-separated):'))
        layout.addWidget(self.channels2use_line)

        # well(s) to use as reference for normalisation using its percentiles
        self.ref_wells_line = QLineEdit()
        layout.addWidget(QLabel('Well(s) to use as reference for normalisation using its percentiles (comma-separated, optional):'))
        layout.addWidget(self.ref_wells_line)

        # indicate that image is a mask. flat field correction will not be applied
        self.masks_cb = QCheckBox('Indicate that image is a mask, flat field correction will not be applied')
        layout.addWidget(self.masks_cb)

        # select path to file to image to use for flat field correction
        self.ff_path_btn = QPushButton('Select inmage file to use for flat field correction (optional)')
        self.ff_path_btn.clicked.connect(self.select_ff_image)
        self.ff_path_label = QLabel('No image selected')
        layout.addWidget(self.ff_path_btn)
        layout.addWidget(self.ff_path_label)

        # colour map to apply to image borders in grid plot
        self.cmap_line = QLineEdit('cet_glasbey')
        layout.addWidget(QLabel('*Colour map to use for boxplots:'))
        layout.addWidget(self.cmap_line)

        # variables to use for colour encoding
        self.condition_vars_line = QLineEdit('Treat,Dose')
        layout.addWidget(QLabel('*Variables to use for colour encoding (comma-separated):'))
        layout.addWidget(self.condition_vars_line)

        # list of condition combinations to remove from the palette
        self.conditions2remove_line = QLineEdit()
        layout.addWidget(QLabel('List of condition combinations to remove from the palette (comma-separated, optional):'))
        layout.addWidget(self.conditions2remove_line)

        # check if input path contains subdirectories
        self.check_batches_cb = QCheckBox('Check if input path contains subdirectories')
        layout.addWidget(self.check_batches_cb)

        # index of field to plot for each well. if none, random field will be selected
        self.field_plot_line = QLineEdit()
        layout.addWidget(QLabel('Index of field to plot (optional):'))
        layout.addWidget(self.field_plot_line)

        # plate ID to use; overwrites plate_layout automatically
        self.plate_id_line = QLineEdit()
        layout.addWidget(QLabel('Plate ID to use.\nOverwrites plate_layout automatically.\nUseful if plate ID is not in the filename. (optional):'))
        layout.addWidget(self.plate_id_line)

        # rep ID to use; overwrites plate_layout automatically
        self.rep_id_line = QLineEdit()
        layout.addWidget(QLabel('Rep ID to use.\nOverwrites plate_layout automatically.\nUseful if rep ID is not in the filename. (optional):'))
        layout.addWidget(self.rep_id_line)

        layout.addWidget(QLabel('Required fields.*'))

        # run or done button to begin processing
        self.run_btn = QPushButton('Visualise Grid Plot')
        self.run_btn.clicked.connect(self.visualise_grid_plot)
        layout.addWidget(self.run_btn)

        self.back_btn = QPushButton("Back")
        self.back_btn.clicked.connect(back_callback)
        layout.addWidget(self.back_btn)
        self.setLayout(layout)

    # specifies to load in the files and folder
    def select_input_folder(self):
        input_folder = QFileDialog.getExistingDirectory(self, 'Select input folder')
        if input_folder:
            self.input_path_label.setText(input_folder)

    def select_output_folder(self):
        output_folder = QFileDialog.getExistingDirectory(self, 'Select output folder')
        if output_folder:
            self.output_path_label.setText(output_folder)

    def select_plate_layout(self):
        plate_layout, _ = QFileDialog.getOpenFileName(self, 'Select file containing plate layout')
        if plate_layout:
            self.plate_layout_label.setText(plate_layout)

    def select_ff_image(self):
        ff_image, _ = QFileDialog.getOpenFileName(self, 'Select image file for flat field correction')
        if ff_image:
            self.ff_path_label.setText(ff_image)

    # gathering all inputs for visualising grid plot
    def visualise_grid_plot(self):

        input_path = self.input_path_label.text()
        output_path = self.output_path_label.text()
        plate_layout = self.plate_layout_label.text()
        suffix = self.suffix_line.text()
        cmap = self.cmap_line.text()
        masks = self.masks_cb.isChecked()
        check_batches = self.check_batches_cb.isChecked()
        img_type = self.img_type_combo.currentText()
        
        # split at ','
        condition_vars = self.condition_vars_line.text().split(',')
        
        # optional variables
        ff_path = self.ff_path_label.text() if self.ff_path_label.text() else None
        plate_id = self.plate_id_line.text() if self.plate_id_line.text() else None
        rep_id = self.rep_id_line.text() if self.rep_id_line.text() else None

        # only split at ',' if text exists
        skip_wells = self.skip_wells_line.text().split(',') if self.skip_wells_line.text() else None
        ref_wells = self.ref_wells_line.text().split(',') if self.ref_wells_line.text() else None
        conditions2remove = self.conditions2remove_line.text().split(',') if self.conditions2remove_line.text() else None

        # convert strings to integers
        well_idx = [int(x) for x in self.well_idx_line.text().split(',')]
        field_idx = [int(x) for x in self.field_idx_line.text().split(',')]
        channels2use = [int(x) for x in self.channels2use_line.text().split(',')] if self.channels2use_line.text() else 0
        field_plot = [int(x) for x in self.field_plot_line.text()] if self.field_plot_line.text() else None

        try:
            input_path = Path(input_path)
            output_path = Path(output_path)
            output_path.mkdir(parents=True, exist_ok=True)

            if isinstance(skip_wells, str):
                skip_wells = [skip_wells]

            if isinstance(condition_vars, str):
                condition_vars = [condition_vars]
            
            if conditions2remove is not None:
                conditions2remove = [tuple(condition.split(',')) for condition in conditions2remove]

            if isinstance(field_plot, int):
                field_plot = [field_plot]
            for idx in field_plot:
                assert idx >= 0, 'field_plot must be a positive integer'

            # Set color palette for treatments
            plate_layout = pd.read_csv(plate_layout, sep=",|;", engine='python'); 
            plate_layout.columns = [col.capitalize() for col in plate_layout.columns]
            plate_layout['Plate'] = plate_layout['Plate'].astype(str)
            plate_layout['Well'] = plate_layout['Well'].astype(str)
            plate_layout['Row'] = plate_layout['Row'].astype(str)
            plate_layout['Col'] = plate_layout['Col'].astype(int)

            # Set color palette
            palette, _ = create_palette(
                plate_layout,
                condition_vars=condition_vars,
                cmap=cmap,
                conditions2remove=conditions2remove,
            )

            # Get flat field images
            if ff_path is not None:
                flat_field_files = get_flat_field_files(ff_path)
            else:
                flat_field_files = None
            
            # Image path
            input_path = Path(input_path)
            df_images = get_df_images(
                input_path,
                check_batches,
                suffix,
                well_idx,
                field_idx,
                skip_wells,
                plate_id,
                rep_id
            )
            df_images.columns = [col.capitalize() for col in df_images.columns]
            df_images['Plate'] = df_images['Plate'].astype(str)
            df_images['Well'] = df_images['Well'].astype(str)
            df_images['Row'] = df_images['Row'].astype(str)
            df_images['Col'] = df_images['Col'].astype(int)
            df_images['Field'] = df_images['Field'].astype(int)

            # Merge plate layout with df_images
            df_images = link_df2plate_layout(df_images, plate_layout)

            # Plot a single image per row/col combination
            for dir in tqdm(sorted(df_images['Dir'].unique())):
                # Load flat field
                if flat_field_files is not None:
                    flat_field_file = [f for f in flat_field_files if Path(dir).stem in f.stem and 'FF' in f.stem][0]
                    flat_field = tifffile.imread(str(flat_field_file)).astype(float)
                else:
                    flat_field = None

                df_dir = df_images[df_images['Dir'] == dir]

                fig = create_grid_plot(
                    df_dir,
                    flat_field,
                    condition_vars,
                    palette,
                    field_idx=field_idx,
                    img_type=img_type,
                    channels2use=channels2use,
                    ref_wells=ref_wells,
                    title=dir
                )

                # Save figure
                filename = Path(output_path).joinpath(f'{dir}.jpg')
                filename.parent.mkdir(exist_ok=True, parents=True)
                print(f'Saving figure to {filename}')
                fig.savefig(filename, dpi=600)
                plt.close()

            QMessageBox.information(self, 'Success', 'Operation completed successfully!')

        except Exception as e:
            QMessageBox.critical(self, 'Error', str(e))

"""
=============================================================
                        CHANNEL PLOT
=============================================================
several parameters/options to customise programme settings:
- select input file
- select output folder
- type of image to plot
- channels to use for making the plots
- suffix of image files
* normalise the images
* well(s) to use as reference for normalisation using its
  percentiles
- start and stop indices of well name in filename
- start and stop indices of field numbers in filename
* select image to use for flat field correction
- percentiles to use for image normalisation
* patterns in filename to ignore
* patterns in filename to include
* index of field to plot
- type of output to save
                                                    optional*
=============================================================
"""
    
class ChannelPlotWindow(QWidget):
    def __init__(self, back_callback):
        super().__init__()
        self.setWindowTitle("Channel Plot")
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Channel Plot Options"))

        # select path to input folder
        self.input_path_btn = QPushButton('*Select input folder')
        self.input_path_btn.clicked.connect(self.select_input_folder)
        self.input_path_label = QLabel('No folder selected')
        layout.addWidget(self.input_path_btn)
        layout.addWidget(self.input_path_label)

        # select path to output folder
        self.output_path_btn = QPushButton('*Select output folder')
        self.output_path_btn.clicked.connect(self.select_output_folder)
        self.output_path_label = QLabel('No folder selected')
        layout.addWidget(self.output_path_btn)
        layout.addWidget(self.output_path_label)

        # type of image to plot
        self.img_type_combo = QComboBox()
        self.img_type_combo.addItems(['grayscale', 'multichannel'])
        layout.addWidget(QLabel('*Type of image to plot'))
        layout.addWidget(self.img_type_combo)

        # channels to use for making the plots
        self.channels2use_line = QLineEdit('0')
        layout.addWidget(QLabel('*Channels to use for making the plots (comma-separated):'))
        layout.addWidget(self.channels2use_line)

        # suffix of image files
        self.suffix_line = QLineEdit('.nd2')
        layout.addWidget(QLabel('*Suffix of image files:'))
        layout.addWidget(self.suffix_line)

        # normalise the images
        self.normalise_cb = QCheckBox('Normalise images')
        layout.addWidget(self.normalise_cb)

        # well(s) to use as reference for normalisation using its percentiles
        self.ref_wells_line = QLineEdit()
        layout.addWidget(QLabel('Well(s) to use as reference for normalisation using its percentiles (comma-separated, optional):'))
        layout.addWidget(self.ref_wells_line)

        # start and stop indices of well name in filename
        self.well_idx_line = QLineEdit('4,7')
        layout.addWidget(QLabel('*Indices of the well name in the filename (start,stop):'))
        layout.addWidget(self.well_idx_line)

        # start and stop indices of field numbers in filename
        self.field_idx_line = QLineEdit('17,21')
        layout.addWidget(QLabel('*Indices of field numbers in filename (start,stop):'))
        layout.addWidget(self.field_idx_line)

        # select path to image to use for flat field correction
        self.select_img_btn = QPushButton('Select image file for flat field correction (optional)')
        self.select_img_btn.clicked.connect(self.select_img_file)
        self.select_img_label = QLabel('No image selected')
        layout.addWidget(self.select_img_btn)
        layout.addWidget(self.select_img_label)

        # percentiles to use for image normalisation
        self.percentiles_line = QLineEdit('0.1,99.9')
        layout.addWidget(QLabel('*Percentiles to use for image normalisation:'))
        layout.addWidget(self.percentiles_line)

        # patterns in filename to ignore
        self.patterns2ignore_line = QLineEdit()
        layout.addWidget(QLabel('Patterns in filename to ignore (comma-separated, optional):'))
        layout.addWidget(self.patterns2ignore_line)

        # patterns in filename to include
        self.patterns2have_line = QLineEdit()
        layout.addWidget(QLabel('Patterns in filename to include (comma-separated, optional):'))
        layout.addWidget(self.patterns2have_line)

        # index of field to plot
        self.field_plot_line = QLineEdit()
        layout.addWidget(QLabel('Index of field to plot (optional):'))
        layout.addWidget(self.field_plot_line)

        # type of output to save
        self.output_type_line = QLineEdit('channels,composite')
        layout.addWidget(QLabel('*Type of output to save:'))
        layout.addWidget(self.output_type_line)

        layout.addWidget(QLabel('Required fields.*'))

        # run or done button to begin processing
        self.run_btn = QPushButton('Visualise Channel Plot')
        self.run_btn.clicked.connect(self.visualise_channel_plot)
        layout.addWidget(self.run_btn)

        self.back_btn = QPushButton("Back")
        self.back_btn.clicked.connect(back_callback)
        layout.addWidget(self.back_btn)
        self.setLayout(layout)

    # specifies to load in the file(s) and folder
    def select_input_folder(self):
        input_folder = QFileDialog.getExistingDirectory(self, 'Select input folder')
        if input_folder:
            self.input_path_label.setText(input_folder)

    def select_output_folder(self):
        output_folder = QFileDialog.getExistingDirectory(self, 'Select output folder')
        if output_folder:
            self.output_path_label.setText(output_folder)

    def select_img_file(self):
        image_file, _ = QFileDialog.getOpenFileName(self, 'Select image file')
        if image_file:
            self.select_img_label.setText(image_file)

    # gathering all inputs for visualising channel plot
    def visualise_channel_plot(self):

        input_path = self.input_path_label.text()
        output_path = self.output_path_label.text()
        suffix = self.suffix_line.text()
        normalise = self.normalise_cb.isChecked()
        img_type = self.img_type_combo.currentText()

        # split at ','
        output_type = self.output_type_line.text().split(',')
        
        # optional variables
        ff_path = self.select_img_label.text() if self.select_img_label.text() else None

        # only split at ',' if text exists
        ref_wells = self.ref_wells_line.text().split(',') if self.ref_wells_line.text() else None
        patterns2ignore = self.patterns2ignore_line.text().split(',') if self.patterns2ignore_line.text() else None
        patterns2have = self.patterns2have_line.text().split(',') if self.patterns2have_line.text() else None

        # convert strings to integers
        well_idx = [int(x) for x in self.well_idx_line.text().split(',')]
        field_idx = [int(x) for x in self.field_idx_line.text().split(',')]
        percentiles = [float(x) for x in self.percentiles_line.text().split(',')] if self.percentiles_line.text() else None
        field_plot = int(self.field_plot_line.text()) if self.field_plot_line.text() else None
        channels2use = [int(x) for x in self.channels2use_line.text().split(',')] if self.channels2use_line.text() else 0
        
        try:
            if ref_wells is not None:
                if isinstance(ref_wells, str):
                    ref_wells = (ref_wells,)
                else:
                    ref_wells = tuple(str(well) for well in ref_wells)

            if patterns2ignore is not None:
                if isinstance(patterns2ignore, str):
                    patterns2ignore = (patterns2ignore,)

            if patterns2have is not None:
                if isinstance(patterns2have, str):
                    patterns2have = (patterns2have,)
            
            if output_type is not None:
                if isinstance(output_type, str):
                    output_type = (output_type,)

            # Call the function
            create_channel_plots(
                input_path=input_path,
                output_path=output_path,
                img_type=img_type,
                channels2use=channels2use,
                suffix=suffix,
                normalize=normalise,
                ref_wells=ref_wells,
                filename_well_idx=well_idx,
                filename_field_idx=field_idx,
                flat_field_path=ff_path,
                percentiles=percentiles,
                pattern2ignore=patterns2ignore,
                patterns2have=patterns2have,
                field_idx=field_plot,
                output_type=output_type
            )

            QMessageBox.information(self, 'Success', 'Operation completed successfully!')

        except Exception as e:
            QMessageBox.critical(self, 'Error', str(e))

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SysMicroPyTools GUI")
        self.menu = MainMenu()

        self.grid2table_window = Grid2TableWindow(self.show_menu)
        self.measuredoseresponse_window = MeasureDoseResponseWindow(self.show_menu)
        self.visualise_window = VisualiseWindow(self.show_menu)
        self.count_window = CountPlotWindow(self.show_visualise)
        self.grid_window = GridPlotWindow(self.show_visualise)
        self.channel_window = ChannelPlotWindow(self.show_visualise)

        self.menu.grid2table_btn.clicked.connect(self.show_grid2table)
        self.menu.measure_dose_response_btn.clicked.connect(self.show_measure_dose_response)
        self.menu.visualise_btn.clicked.connect(self.show_visualise)
        self.visualise_window.countplot_btn.clicked.connect(self.show_count)
        self.visualise_window.gridplot_btn.clicked.connect(self.show_grid)
        self.visualise_window.channelplot_btn.clicked.connect(self.show_channel)

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.menu)
        self.layout.addWidget(self.grid2table_window)
        self.layout.addWidget(self.measuredoseresponse_window)
        self.layout.addWidget(self.visualise_window)
        self.layout.addWidget(self.count_window)
        self.layout.addWidget(self.grid_window)
        self.layout.addWidget(self.channel_window)
        self.setLayout(self.layout)

        self.show_menu()

    def show_menu(self):
        self.menu.show()
        self.grid2table_window.hide()
        self.measuredoseresponse_window.hide()
        self.visualise_window.hide()
        self.count_window.hide()
        self.grid_window.hide()
        self.channel_window.hide()

    def show_grid2table(self):
        self.menu.hide()
        self.grid2table_window.show()

    def show_measure_dose_response(self):
        self.menu.hide()
        self.measuredoseresponse_window.show()

    def show_visualise(self):
        self.menu.hide()
        self.visualise_window.show()
        self.count_window.hide()
        self.grid_window.hide()
        self.channel_window.hide()

    def show_count(self):
        self.menu.hide()
        self.visualise_window.hide()
        self.count_window.show()

    def show_grid(self):
        self.menu.hide()
        self.visualise_window.hide()
        self.grid_window.show()

    def show_channel(self):
        self.menu.hide()
        self.visualise_window.hide()
        self.channel_window.show()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())