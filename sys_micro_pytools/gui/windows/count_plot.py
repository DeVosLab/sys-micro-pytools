"""
VISUALISING COUNT PLOT WINDOW: defines the window for the tool to visualise count plot
o   this script contains all UI elements for the tool and
o   accepts a callback to return to the previous page
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from PySide6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel, QLineEdit, QFileDialog, QCheckBox, QMessageBox, QComboBox

from sys_micro_pytools.df import link_df2plate_layout
from sys_micro_pytools.visualize import create_palette
from sys_micro_pytools.visualize.grid_plots import get_df_images
from sys_micro_pytools.visualize.count_plots import create_count_df, create_count_plot

class CountPlotWindow(QWidget):
    """Window for count plot visualisation tool."""

    def __init__(self, back_callback):
        super().__init__()
        self.setWindowTitle("Count Plot")
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Count Plot Options"))

        ## COUNT PLOT OPTIONS BELOW ##

        # select path to input file
        self.input_path_btn = QPushButton('Select input file')
        self.input_path_btn.clicked.connect(self.select_input_file)
        self.input_path_label = QLabel('No file selected')
        layout.addWidget(self.input_path_btn)
        layout.addWidget(self.input_path_label)        

        # select path to output folder
        self.output_path_btn = QPushButton('Select output folder')
        self.output_path_btn.clicked.connect(self.select_output_folder)
        self.output_path_label = QLabel('No folder selected')
        layout.addWidget(self.output_path_btn)
        layout.addWidget(self.output_path_label)

        # select path to file containing plate layout
        self.plate_layout_btn = QPushButton('Select file containing plate layout')
        self.plate_layout_btn.clicked.connect(self.select_layout_file)
        self.plate_layout_label = QLabel('No file selected')
        layout.addWidget(self.plate_layout_btn)
        layout.addWidget(self.plate_layout_label)

        # suffix of image files
        self.suffix_line = QLineEdit('.tif')
        layout.addWidget(QLabel('Suffix of image files:'))
        layout.addWidget(self.suffix_line)

        # start and stop indices of well name in filename
        self.well_idx_line = QLineEdit('4,7')
        layout.addWidget(QLabel('Indices of well name in filename (start,stop):'))
        layout.addWidget(self.well_idx_line)

        # start and stop indices of field number in filename
        self.field_idx_line = QLineEdit('17,21')
        layout.addWidget(QLabel('Indices of field number in filename (start,stop):'))
        layout.addWidget(self.field_idx_line)

        # list of wells to skip
        self.skip_wells_line = QLineEdit()
        layout.addWidget(QLabel('List of wells to skip (comma-separated, optional):'))
        layout.addWidget(self.skip_wells_line)

        # colour map to use for boxplots
        self.cmap_line = QLineEdit('cet_glasbey')
        layout.addWidget(QLabel('Colour map to use for boxplots:'))
        layout.addWidget(self.cmap_line)

        # variables to use for colour encoding
        self.condition_vars_line = QLineEdit('Treat,Dose')
        layout.addWidget(QLabel('Variables to use for colour encoding (comma-separated):'))
        layout.addWidget(self.condition_vars_line)

        # list of condition combinations to remove from the palette
        self.conditions2remove_line = QLineEdit()
        layout.addWidget(QLabel('List of condition combinations to remove from the palette (comma-separated, optional):'))
        layout.addWidget(self.conditions2remove_line)

        # check if input path contains subdirectories
        self.check_batches_cb = QCheckBox('Check if input path contains subdirectories')
        layout.addWidget(self.check_batches_cb)

        # label for y-axis
        self.y_label_line = QLineEdit('Object Count')
        layout.addWidget(QLabel('Label for y-axis:'))
        layout.addWidget(self.y_label_line)

        # limits for y-axis
        self.y_lim_line = QLineEdit()
        layout.addWidget(QLabel('Limits for y-axis (min,max; optional):'))
        layout.addWidget(self.y_lim_line)

        # width of boxes in the boxplot
        self.box_width_line = QLineEdit('0.8')
        layout.addWidget(QLabel('Width of boxes in the boxplot:'))
        layout.addWidget(self.box_width_line)

        # add jitter to the points
        self.jitter_cb = QCheckBox('Add jitter to the points')
        layout.addWidget(self.jitter_cb)

        # type of plot to create
        self.plot_type_combo = QComboBox()
        self.plot_type_combo.addItems(['box', 'violin', 'swarm'])
        layout.addWidget(QLabel('Type of plot to create:'))
        layout.addWidget(self.plot_type_combo)

        # format for output figures
        self.output_format_line = QLineEdit('png')
        layout.addWidget(QLabel('Format for output figures:'))
        layout.addWidget(self.output_format_line)

        # DPI for output figures
        self.dpi_line = QLineEdit('150')
        layout.addWidget(QLabel('DPI for output figures:'))
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

        # run or done button to begin processing
        self.run_btn = QPushButton('Visualise Count Plot')
        self.run_btn.clicked.connect(self.visualise_count_plot)
        layout.addWidget(self.run_btn)

        self.back_btn = QPushButton("Back")
        self.back_btn.clicked.connect(back_callback)
        layout.addWidget(self.back_btn)
        self.setLayout(layout)

    ## LOADING IN FILES & FOLDERS ##

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

    ## GATHERING AND PROCESSING INPUT ##

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