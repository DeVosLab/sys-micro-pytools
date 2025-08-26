import sys

from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton, QFileDialog, QCheckBox, QSpinBox, QMessageBox
from sys_micro_pytools.df.plate_grid2table import plate_grid2table, plot_layout

class PlateGrid2TableGUI(QWidget):
    # initialise GUI
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Plate Grid 2 Table GUI')
        self.layout = QVBoxLayout()

        # select path to input file
        self.input_path_btn = QPushButton('Select Input File')
        self.input_path_btn.clicked.connect(self.select_input_file)
        self.input_path_label = QLabel('No file selected')
        self.layout.addWidget(self.input_path_btn)
        self.layout.addWidget(self.input_path_label)

        # select path to save the merged layout dataframe
        # if not specified, output will be saved in same directory as input file(s)
        self.output_path_btn = QPushButton('Select Output Folder')
        self.output_path_btn.clicked.connect(self.select_output_folder)
        self.output_path_label = QLabel('No folder selected')
        self.layout.addWidget(self.output_path_btn)
        self.layout.addWidget(self.output_path_label)
        
        # write name of output file
        self.filename_edit = QLineEdit()
        self.layout.addWidget(QLabel('Output Filename (.csv):'))
        self.layout.addWidget(self.filename_edit)

        # check box to visualise plate layout
        self.visualize_cb = QCheckBox('Visualise Plate Layout')
        self.layout.addWidget(self.visualize_cb)

        # index of plate to be visualised
        # if not specified, all plates will be visualised
        self.plate_id_edit = QLineEdit()
        self.layout.addWidget(QLabel('Plate ID (comma-separated, optional):'))
        self.layout.addWidget(self.plate_id_edit)

        # categories for row variable, i.e. row names
        self.row_categories_edit = QLineEdit('A,B,C,D,E,F,G,H')
        self.layout.addWidget(QLabel('Row categories (comma-separated):'))
        self.layout.addWidget(self.row_categories_edit)

        # categories for column variable, i.e. column names
        self.col_categories_edit = QLineEdit('1,2,3,4,5,6,7,8,9,10,11,12')
        self.layout.addWidget(QLabel('Column categories (comma-separated):'))
        self.layout.addWidget(self.col_categories_edit)

        # order of variables in the visualisation
        self.var_order_edit = QLineEdit()
        self.layout.addWidget(QLabel('Order of variables (comma-separated, optional):'))
        self.layout.addWidget(self.var_order_edit)

        # number of columns for the subplots in the visualisation
        self.ncols_spin = QSpinBox()
        self.ncols_spin.setMinimum(1)
        self.ncols_spin.setValue(3)
        self.layout.addWidget(QLabel('Number of columns for subplots:'))
        self.layout.addWidget(self.ncols_spin)

        # check box to add annotations to the heatmap
        self.add_annot_cb = QCheckBox('Add annotations to heatmap')
        self.layout.addWidget(self.add_annot_cb)

        # variables to be treated as numeric in the visualisation
        self.numeric_vars_edit = QLineEdit()
        self.layout.addWidget(QLabel('Numeric variables (comma-separated, optional):'))
        self.layout.addWidget(self.numeric_vars_edit)

        # check box to remove rows containing NA values from the dataframe
        self.remove_na_cb = QCheckBox('Remove rows with NA')
        self.layout.addWidget(self.remove_na_cb)

        # run or done button to begin processing
        self.run_btn = QPushButton('Run')
        self.run_btn.clicked.connect(self.run_plate_grid2table)
        self.layout.addWidget(self.run_btn)

        self.setLayout(self.layout)

    # specifies to load in a file
    def select_input_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, 'Select Input File')
        if file_path:
            self.input_path_label.setText(file_path)

    # specifies to load in a folder
    def select_output_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, 'Select Output Folder')
        if folder_path:
            self.output_path_label.setText(folder_path)

    # gathering all inputs
    def run_plate_grid2table(self):

        input_path = self.input_path_label.text()
        ncols = self.ncols_spin.value()
        visualize = self.visualize_cb.isChecked()
        add_annot = self.add_annot_cb.isChecked()
        remove_rows_with_na = self.remove_na_cb.isChecked()

        # optional variables
        output_path = self.output_path_label.text() or None
        filename = self.filename_edit.text() or None
        plate_id = self.plate_id_edit.text() or None

        # split at ','
        row_categories = self.row_categories_edit.text().split(',')
        # convert strings to integers
        col_categories = [int(x) for x in self.col_categories_edit.text().split(',')]

        # optional variables, only split at ',' if text exists
        var_order = self.var_order_edit.text().split(',') if self.var_order_edit.text() else None
        numeric_vars = self.numeric_vars_edit.text().split(',') if self.numeric_vars_edit.text() else None

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

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = PlateGrid2TableGUI()
    window.show()
    sys.exit(app.exec())