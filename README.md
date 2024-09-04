[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![image](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lbferreira/geocoreg/blob/main)
# geocoreg
A library to simplify the process of co-registering geospatial data.
Althought some great libraries provide co-registration functionalities, such as scikit-image and kornia, they don't directly support geospatial data. This library aims to provide a simple and easy-to-use interface to co-register geospatial data in a pythonic way. The library design was created to allow the user to select different co-registration methods from different back-end libraries. For an easier utilization, Xarray DataArrays are supported (dask is not supported yet).

## Installation
To install the package, run the command below
```
pip install git+https://github.com/lbferreira/geocoreg
```

## Usage
To coregister a single image or multiple images (e.g., a time series) in relation to a reference image, you can use the `coregister` function, as shown below.
```python
from geocoreg import xr_coregistration as xcr

image_to_coregister = ... # Xarray DataArray it can be a single or multiple images
reference_image = ... # Xarray DataArray
coregistrated_image = xcr.coregistrate(image_to_coregister, reference_image, registration_bands=['red'])
```

## Acknowledgements
This library was developed as part of
my research work in the [GCER lab](https://www.gcerlab.com/), under supervision of Vitor Martins, at the Mississippi State University (MSU). This research is funded by USDA NIFA (award #2023-67019-39169), supporting Lucas Ferreira and Vitor Martins at MSU.