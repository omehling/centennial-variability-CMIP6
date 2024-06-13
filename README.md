# centennial-variability-CMIP6

This repository provides Jupyter notebooks and analysis code for the paper

O. Mehling, K. Bellomo, J. von Hardenberg (2024): Centennial-scale variability of the Atlantic Meridional Circulation in CMIP6 models shaped by Arctic–North Atlantic interactions and sea ice biases, submitted.

## Overview

The main directory contains Jupyter notebook to reproduce figures from the main text and supplementary material:

* `Plots_AMOC_strength.ipynb` for Fig. 1 and Supplementary Figs. S2–S4
* `Plots_Arctic_freshwater.ipynb` for Fig. 2 and Supplementary Figs. S1, S5–S7
* `Plots_Sea_ice.ipynb` for Fig. 3 and Supplementary Figs. S8–S9

Python functions used in the analysis are bundled in `scripts`. Pre-processing scripts for Arctic Ocean mean salinity, freshwater content and gateway transports from standard CMIP6 output can be found in `preprocessing`.

## Requirements
```
numpy
scipy
pandas
xarray
nitime
eofs
matplotlib
cartopy
seaborn
cmocean
```

Additional requirements for preprocessing:
```
regionmask
haversine
dask
```