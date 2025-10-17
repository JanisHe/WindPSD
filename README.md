# Wind Dependent PSD spectra

## Installation
Either download this repository to work in the project or install this project by running
`pip install git+https://github.com/JanisHe/WindPSD`

## Requirements
`numpy`, `obspy`, `tqdm`, `meteostat` (https://dev.meteostat.net/python/)

## Setting parameters
In `parfiles/parfile.yml` you can find an example for settings. This parameter file is loaded
in `main`, however, instead of the parameter file the function `main` also takes a dictionary
as input. When working with a dictionary, the keys are the same keys as in `parfiles/parfile.yml`.

## Compute Spectra
Once you have created or modified your `parfile.yml`, you can start the computation of the
spectra by running the `main` function from `windpsd.main`:
```python
from windpsd.main import main

parfile = "./parfiles/parfile.yml"
main(parfile=parfile)
```

## Possible issues
- In case you get a ModuleImportError try to add the PythonPath to your project by
```python
import sys
sys.path.append("/path/to/my/windpsd/project")
```
- Others errors are might be raised by the main function. Then please check if your settings are correct.

## Citation
- Heuel, J., & Friederich, W. (2022). Suppression of wind turbine noise from seismological data using nonlinear thresholding and denoising autoencoder. Journal of Seismology, 26(5), 913-934.
- Stammler, K., & Ceranna, L. (2016). Influence of wind turbines on seismic records of the Gräfenberg array. Seismological Research Letters, 87(5), 1075-1081.
