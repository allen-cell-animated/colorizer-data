# colorizer-data
Example scripts and utilities for preparing data for the time series colorizer app

Data must be preprocessed to work with Nucmorph-Colorizer. [You can read more about the data format specification here.](./documentation/DATA_FORMAT.md)

Utilities are included in this repository to convert time-series data.  We provide utilities for writing data to our specification, and custom scripts for some AICS internal projects' data sets. Each project (AICS nuclear morphogenesis, AICS EMT, etc) will need their own data conversion scripts.

For loading of datasets to work correctly, you'll need to run these commands from a device that has access to Allen Institute's on-premises data storage. If running off of shared resources, remember to initialize your virtual environment first! This may look like `conda activate {my_env}`.

NOTE: You may also need to run Python version 3.8, due to dependencies in the [Nucmorph Analysis package](https://github.com/aics-int/nuc-morph-analysis/blob/main/docs/INSTALL.md#basic-installation-instructions-with-conda-and-pip) this project was originally built for. For conda, you can create an environment with `conda create --name {my_env} python=3.8`.

In order to use the `data_writer_utils` as a Python package, install the Python dependencies:

```
pip install .
```

The `convert_nucmorph_data.py` script can take in a named Nuclear Morphogenesis dataset (like `baby_bear`, `mama_bear`, or `goldilocks`) and convert it to a format readable
by the web client.

```
python timelapse-colorizer-data/convert_nucmorph_data.py --output_dir {output_dir} --dataset {dataset_name} --scale 0.25
```