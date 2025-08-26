import sys

from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QLineEdit, QFileDialog, QCheckBox, QSpinBox, QMessageBox

from sys_micro_pytools.df.plate_grid2table import plate_grid2table, plot_layout

# main menu / home page
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

class Grid2TableWindow(QWidget):
    def __init__(self, back_callback):
        super().__init__()
        self.setWindowTitle("Grid2Table")
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Grid2Table Options"))

        # select path to input file
        self.input_path_btn = QPushButton('Select Input File')
        self.input_path_btn.clicked.connect(self.select_input_file)
        self.input_path_label = QLabel('No file selected')
        layout.addWidget(self.input_path_btn)
        layout.addWidget(self.input_path_label)

        # select path to save the merged layout dataframe
        # if not specified, output will be saved in same directory as input file(s)
        self.output_path_btn = QPushButton('Select Output Folder')
        self.output_path_btn.clicked.connect(self.select_output_folder)
        self.output_path_label = QLabel('No folder selected')
        layout.addWidget(self.output_path_btn)
        layout.addWidget(self.output_path_label)
        
        # write name of output file
        self.filename_edit = QLineEdit()
        layout.addWidget(QLabel('Output Filename (.csv):'))
        layout.addWidget(self.filename_edit)

        # check box to visualise plate layout
        self.visualize_cb = QCheckBox('Visualise Plate Layout')
        layout.addWidget(self.visualize_cb)

        # index of plate to be visualised
        # if not specified, all plates will be visualised
        self.plate_id_edit = QLineEdit()
        layout.addWidget(QLabel('Plate ID (comma-separated, optional):'))
        layout.addWidget(self.plate_id_edit)

        # categories for row variable, i.e. row names
        self.row_categories_edit = QLineEdit('A,B,C,D,E,F,G,H')
        layout.addWidget(QLabel('Row categories (comma-separated):'))
        layout.addWidget(self.row_categories_edit)

        # categories for column variable, i.e. column names
        self.col_categories_edit = QLineEdit('1,2,3,4,5,6,7,8,9,10,11,12')
        layout.addWidget(QLabel('Column categories (comma-separated):'))
        layout.addWidget(self.col_categories_edit)

        # order of variables in the visualisation
        self.var_order_edit = QLineEdit()
        layout.addWidget(QLabel('Order of variables (comma-separated, optional):'))
        layout.addWidget(self.var_order_edit)

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
        self.numeric_vars_edit = QLineEdit()
        layout.addWidget(QLabel('Numeric variables (comma-separated, optional):'))
        layout.addWidget(self.numeric_vars_edit)

        # check box to remove rows containing NA values from the dataframe
        self.remove_na_cb = QCheckBox('Remove rows with NA')
        layout.addWidget(self.remove_na_cb)

        # run or done button to begin processing
        self.run_btn = QPushButton('Run')
        self.run_btn.clicked.connect(self.run_plate_grid2table)
        layout.addWidget(self.run_btn)

        self.back_btn = QPushButton("Back")
        self.back_btn.clicked.connect(back_callback)
        layout.addWidget(self.back_btn)
        self.setLayout(layout)

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

    # gathering all inputs for grid2table
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

class MeasureDoseResponseWindow(QWidget):
    def __init__(self, back_callback):
        super().__init__()
        self.setWindowTitle("Measure Dose Response")
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Options to Measure Dose Response Here"))
        self.back_btn = QPushButton("Back")
        self.back_btn.clicked.connect(back_callback)
        layout.addWidget(self.back_btn)
        self.setLayout(layout)

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

class CountPlotWindow(QWidget):
    def __init__(self, back_callback):
        super().__init__()
        self.setWindowTitle("Count Plot")
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Count Plot Options Here"))
        self.back_btn = QPushButton("Back")
        self.back_btn.clicked.connect(back_callback)
        layout.addWidget(self.back_btn)
        self.setLayout(layout)

class GridPlotWindow(QWidget):
    def __init__(self, back_callback):
        super().__init__()
        self.setWindowTitle("Grid Plot")
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Grid Plot Options Here"))
        self.back_btn = QPushButton("Back")
        self.back_btn.clicked.connect(back_callback)
        layout.addWidget(self.back_btn)
        self.setLayout(layout)

class ChannelPlotWindow(QWidget):
    def __init__(self, back_callback):
        super().__init__()
        self.setWindowTitle("Channel Plot")
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Channel Plot Options Here"))
        self.back_btn = QPushButton("Back")
        self.back_btn.clicked.connect(back_callback)
        layout.addWidget(self.back_btn)
        self.setLayout(layout)

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

    def show_count(self):
        self.menu.hide()
        self.count_window.show()

    def show_grid(self):
        self.menu.hide()
        self.grid_window.show()

    def show_channel(self):
        self.menu.hide()
        self.channel_window.show()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())