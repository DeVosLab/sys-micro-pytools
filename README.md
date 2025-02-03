# sys-micro-pytools

Utilities for data pre/post-processing in Python

## Installation
There are two recommended ways to install `sys-micro-pytools`:

### Option 1: Using conda/mamba (recommended)
This ensures that all dependencies are installed in a compatible environment.

Clone the repository and navigate to its root directory
```bash
git clone https://github.com/tim-vdl/sys-micro-pytools.git
cd sys-micro-pytools
```

Create and activate the environment
```bash
conda env create -f environment.yml
conda activate sys-micro-pytools
```

Install the package
```bash
pip install -e .
```

### Option 2: Using pip
If you manage your own Python environment, you can install the package using pip.

```bash
pip install git+https://github.com/tim-vdl/sys-micro-pytools.git
```

This will automatically install the following dependencies if they're not present:
- numpy
- pandas
- matplotlib
- seaborn
- tifffile
- nd2reader
- scikit-image
- colorcet
- tqdm
- pyvista

**Note**: Some scientific packages might work better when installed through conda. If you experience issues with the pip installation, please try the conda method instead.

## Features

- IO utilities for handling microscopy data
- Pre-processing tools
  - Flat-field correction
  - Image normalization using percentiles
  - Composite image creation
  - Plate grid to layout conversion
- Visualization tools
  - Channel plots
  - Grid plots
- Rendering utilities

## Usage

Basic example of how to use the package:

### Flat-field correction and composite image creation:
```python
from sys_micro_pytools.io import read_tiff_or_nd2
from sys_micro_pytools.preprocess import flat_field_correction, create_composite

# Load image
img = read_tiff_or_nd2('path/to/your/data.tif') # Let's img.shape = (5, 512, 512)

# Load flat-field correction image
flat_field = read_tiff_or_nd2('path/to/your/flat_field.tif') # Also flat_field.shape = (512, 512)

# Pre-process data
img = flat_field_correction(img, flat_field)
img = create_composite(img, channel_dim=0, normalize=True, clip=True)

# Save image
tifffile.imwrite('path/to/your/processed_image.tif', img)
```

### Convert a plate layout as a list in a .csv file
Create a plate layout as a list in a .csv file from a plate layout grid provided in a .csv or .xlsx file(s). Visualize the plate layouts for each identified plate id. Different plates can be provided as separate files (.csv or .xlsx), or as separate sheets in a single .xlsx file. The name of the plate id is extracted from the filename or sheet name using the pattern "_plateX" where X is an alphanumeric character or sequence of such characters.
```bash
python sys_micro_pytools/preprocess/plate_grid2layout.py -i path/to/your/plate_layout.xlsx -o path/to/your/plate_layout.csv -v
```

### Grid plots
Store grid plots of multichannel images according to a plate layout. Use `--hue_vars` to specify the variables to use for color encoding.
```bash
python sys_micro_pytools/visualize/grid_plots.py -i path/to/your/data -o path/to/your/output --plate_layout path/to/your/plate_layout.csv --hue_vars Treat Dose
```

### Channel plots
Store channel plots and composite images of multichannel images.
```bash
python sys_micro_pytools/visualize/channel_plots.py -i path/to/your/data -o path/to/your/output --channels2use 0 1 2 --output_type channels composite
```