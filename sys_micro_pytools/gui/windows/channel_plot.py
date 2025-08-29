"""
VISUALISING CHANNEL PLOT WINDOW: defines the window for the tool to visualise channel plot
o   this script contains all UI elements for the tool and
o   accepts a callback to return to the previous page
"""

from PySide6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel, QLineEdit, QFileDialog, QCheckBox, QMessageBox, QComboBox

from sys_micro_pytools.visualize.channel_plots import create_channel_plots

class ChannelPlotWindow(QWidget):
    def __init__(self, back_callback):
        super().__init__()
        self.setWindowTitle("Channel Plot")
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Channel Plot Options"))

        ## CHANNEL PLOT OPTIONS BELOW ##

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

        # type of image to plot
        self.img_type_combo = QComboBox()
        self.img_type_combo.addItems(['grayscale', 'multichannel'])
        layout.addWidget(QLabel('Type of image to plot'))
        layout.addWidget(self.img_type_combo)

        # channels to use for making the plots
        self.channels2use_line = QLineEdit('0')
        layout.addWidget(QLabel('Channels to use for making the plots (comma-separated):'))
        layout.addWidget(self.channels2use_line)

        # suffix of image files
        self.suffix_line = QLineEdit('.nd2')
        layout.addWidget(QLabel('Suffix of image files:'))
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
        layout.addWidget(QLabel('Indices of the well name in the filename (start,stop):'))
        layout.addWidget(self.well_idx_line)

        # start and stop indices of field numbers in filename
        self.field_idx_line = QLineEdit('17,21')
        layout.addWidget(QLabel('Indices of field numbers in filename (start,stop):'))
        layout.addWidget(self.field_idx_line)

        # select path to image to use for flat field correction
        self.select_img_btn = QPushButton('Select image file for flat field correction (optional)')
        self.select_img_btn.clicked.connect(self.select_img_file)
        self.select_img_label = QLabel('No image selected')
        layout.addWidget(self.select_img_btn)
        layout.addWidget(self.select_img_label)

        # percentiles to use for image normalisation
        self.percentiles_line = QLineEdit('0.1,99.9')
        layout.addWidget(QLabel('Percentiles to use for image normalisation:'))
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
        layout.addWidget(QLabel('Type of output to save:'))
        layout.addWidget(self.output_type_line)

        # run or done button to begin processing
        self.run_btn = QPushButton('Visualise Channel Plot')
        self.run_btn.clicked.connect(self.visualise_channel_plot)
        layout.addWidget(self.run_btn)

        self.back_btn = QPushButton("Back")
        self.back_btn.clicked.connect(back_callback)
        layout.addWidget(self.back_btn)
        self.setLayout(layout)

    ## LOADING IN FILES & FOLDERS ##

    # specifies to load in the file(s) and folder
    def select_input_file(self):
        input_file, _ = QFileDialog.getOpenFileName(self, 'Select input file')
        if input_file:
            self.input_path_label.setText(input_file)

    def select_output_folder(self):
        output_folder, _ = QFileDialog.getExistingDirectory(self, 'Select output folder')
        if output_folder:
            self.output_path_label.setText(output_folder)

    def select_img_file(self):
        image_file, _ = QFileDialog.getOpenFileName(self, 'Select image file')
        if image_file:
            self.select_img_label.setText(image_file)

    ## GATHERING AND PROCESSING INPUT ##

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
        well_idx = int(self.well_idx_line.text().split(','))
        field_idx = int(self.field_idx_line.text().split(','))
        percentiles = float(self.percentiles_line.text().split(',')) if self.percentiles_line.text() else None
        field_plot = int(self.field_plot_line.text()) if self.field_plot_line.text() else None
        channels2use = int(self.channels2use_line.text().split(',')) if self.channels2use_line.text() else 0
        
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