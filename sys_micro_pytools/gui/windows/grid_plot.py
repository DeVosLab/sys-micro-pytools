"""
VISUALISING GRID PLOT WINDOW: defines the window for the tool to visualise grid plot
o   this script contains all UI elements for the tool and
o   accepts a callback to return to the previous page
"""

from pathlib import Path
import tifffile
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from PySide6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel, QLineEdit, QFileDialog, QCheckBox, QMessageBox, QComboBox

from sys_micro_pytools.df import link_df2plate_layout
from sys_micro_pytools.preprocess.flat_field import get_flat_field_files
from sys_micro_pytools.visualize import create_palette
from sys_micro_pytools.visualize.grid_plots import get_df_images, create_grid_plot

class GridPlotWindow(QWidget):
    def __init__(self, back_callback):
        super().__init__()
        self.setWindowTitle("Grid Plot")
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Grid Plot Options"))

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
        self.plate_layout_btn.clicked.connect(self.select_plate_layout)
        self.plate_layout_label = QLabel('No file selected')
        layout.addWidget(self.plate_layout_btn)
        layout.addWidget(self.plate_layout_label)

        # suffix of image files
        self.suffix_line = QLineEdit('.nd2')
        layout.addWidget(QLabel('Suffix of image files:'))
        layout.addWidget(self.suffix_line)

        # start and stop indices of well name in the filename
        self.well_idx_line = QLineEdit('4,7')
        layout.addWidget(QLabel('Indices of well name in filename (start,stop):'))
        layout.addWidget(self.well_idx_line)

        # start and stop indices of field number in the filename
        self.field_idx_line = QLineEdit('17,21')
        layout.addWidget(QLabel('Indices of field number in filename (start,stop):'))
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
        layout.addWidget(QLabel('Channels to use for making the plots (comma-separated):'))
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

        # run or done button to begin processing
        self.run_btn = QPushButton('Visualise Grid Plot')
        self.run_btn.clicked.connect(self.visualise_grid_plot)
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

    def select_plate_layout(self):
        plate_layout, _ = QFileDialog.getOpenFileName(self, 'Select file containing plate layout')
        if plate_layout:
            self.plate_layout_label.setText(plate_layout)

    def select_ff_image(self):
        ff_image, _ = QFileDialog.getOpenFileName(self, 'Select image file for flat field correction')
        if ff_image:
            self.ff_path_label.setText(ff_image)

    ## GATHERING AND PROCESSING INPUT ##

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
        well_idx = int(self.well_idx_line.text().split(','))
        field_idx = int(self.field_idx_line.text().split(','))
        channels2use = int(self.channels2use_line.text().split(',')) if self.channels2use_line.text() else 0
        field_plot = int(self.field_plot_line.text()) if self.field_plot_line.text() else None

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