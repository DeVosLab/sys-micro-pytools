"""
GRID2TABLE WINDOW: defines the window for the Grid2Table tool
o   this script contains all UI elements for the tool and
o   accepts a callback to return to the main menu
"""

from pathlib import Path

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QLabel, QLineEdit,
    QCheckBox, QSpinBox, QFileDialog, QMessageBox)

from sys_micro_pytools.df.plate_grid2table import plate_grid2table, plot_layout

class Grid2TableWindow(QWidget):
    """Window for the Grid2Table tool."""

    def __init__(self, back):
        super().__init__()
        self.setWindowTitle('Grid2Table')
        layout = QVBoxLayout()
        layout.addWidget(QLabel('Grid2Table Options'))

        ## GRID2TABLE OPTIONS BELOW ##

         # select path to input file
        self.input_path_btn = QPushButton('Select input file')
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
        self.visualise_cb = QCheckBox('Visualise plate layout')
        layout.addWidget(self.visualise_cb)

        # index of plate to be visualised
        # if not specified, all plates will be visualised
        self.plate_id_line = QLineEdit()
        layout.addWidget(QLabel('Plate ID (comma-separated, optional):'))
        layout.addWidget(self.plate_id_line)

        # categories for row variable, i.e. row names
        self.row_categories_line = QLineEdit('A,B,C,D,E,F,G,H')
        layout.addWidget(QLabel('Row categories (comma-separated):'))
        layout.addWidget(self.row_categories_line)

        # categories for column variable, i.e. column names
        self.col_categories_line = QLineEdit('1,2,3,4,5,6,7,8,9,10,11,12')
        layout.addWidget(QLabel('Column categories (comma-separated):'))
        layout.addWidget(self.col_categories_line)

        # order of variables in the visualisation
        self.var_order_line = QLineEdit()
        layout.addWidget(QLabel('Order of variables (comma-separated, optional):'))
        layout.addWidget(self.var_order_line)

        # number of columns for the subplots in the visualisation
        self.ncols_spin = QSpinBox()
        self.ncols_spin.setMinimum(1)
        self.ncols_spin.setValue(3)
        layout.addWidget(QLabel('Number of columns for subplots:'))
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

        # run or done button to begin processing
        self.run_btn = QPushButton('Run Grid2Table')
        self.run_btn.clicked.connect(self.run_plate_grid2table)
        layout.addWidget(self.run_btn)

        # back button to return to previous page
        self.back_btn = QPushButton("Back")
        self.back_btn.clicked.connect(back)
        layout.addWidget(self.back_btn)

        self.setLayout(layout)

    ## LOADING IN FILES & FOLDERS ##

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

    ## GATHERING AND PROCESSING INPUT ##

    # gathering all inputs for grid2table
    def run_plate_grid2table(self):

        input_path = self.input_path_label.text()
        ncols = self.ncols_spin.value()
        visualise = self.visualise_cb.isChecked()
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
                # if no filename, use input path as output filename
                filename_path = Path(filename) if filename else Path(input_path).with_suffix('.csv')
                full_path = Path(output_path) / filename_path
                df.to_csv(full_path, index=False)

            if visualise:
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