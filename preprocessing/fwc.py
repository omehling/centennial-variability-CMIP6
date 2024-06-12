#!/usr/bin/env python3
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import pandas as pd
import regionmask
import sys
import glob
import pathlib

from dask_jobqueue import SLURMCluster
from dask.distributed import Client

def layer_weights_all(levels, dtop=0, dbot=None, coord=None):
    if coord is None:
        # try to infer coordinate name from input
        if len(list(levels.coords)) == 1:
            coord = list(levels.coords)[0]
        else:
            print("Warning: Could not infer coordinate name from input, using default: 'level'")
            coord = "level"
    lev = levels.values
    layer_weights_mid = np.zeros(len(lev))
    layer_weights_mid[1:-1] = (lev[2:]-lev[:-2])/2  # size for all layers except upper- and lowermost
    layer_weights_mid[0] = (lev[0]+lev[1])/2 - dtop
    if dbot is None:
        layer_weights_mid[-1] = layer_weights_mid[-2]
    else:
        layer_weights_mid[-1] = dbot - (lev[-1]+lev[-2])/2
    return xr.DataArray(
        data = layer_weights_mid, dims=[coord], coords={coord: levels[coord].values}, name='layer_depths'
        )

def add_encoding(fwcont, so, lat_name, lon_name):
    # Chunk sizes
    chunks_out = []
    size_out = []
    for dim in fwcont.dims:
        dimsize = len(fwcont[dim])
        if dim=="time":
            chunks_out.append(1)
        else:
            chunks_out.append(dimsize)
        size_out.append(dimsize)
    fwcont.encoding = so.encoding
    fwcont.encoding["chunksizes"] = tuple(chunks_out)
    fwcont.encoding["original_shape"] = tuple(size_out)
    # Attributes + encoding for lat/lon
    fwcont[lat_name].attrs = so[lat_name].attrs
    fwcont[lon_name].attrs = so[lon_name].attrs
    fwcont[lat_name].encoding = so[lat_name].encoding
    fwcont[lon_name].encoding = so[lon_name].encoding
    
    return fwcont

# Model name = command line argument
if len(sys.argv) == 1:
    raise TypeError("{} missing 1 required positional argument 'model_id'".format(sys.argv[0]))
elif len(sys.argv) > 2:
    raise TypeError("{} takes 1 positional argument but {} were given".format(sys.argv[0], len(sys.argv)-1))
model_id = str(sys.argv[1])

root_path = "/data/datasets/synda/data/CMIP6/CMIP/"

# Read from config
df_config = pd.read_csv("Model_overview_so.csv", sep=";").set_index("model_id")
if pd.isna(df_config.loc[model_id,"lat_name"]):
    raise KeyError("Model output for {} is not available".format(model_id))
area_path = root_path+df_config.loc[model_id,"area_path"]
so_path = root_path+df_config.loc[model_id,"so_path"]
lat_name = df_config.loc[model_id,"lat_name"]
lon_name = df_config.loc[model_id,"lon_name"]
lev_name = df_config.loc[model_id,"lev_name"]
lon360 = bool(df_config.loc[model_id,"lon360"])

areacello = xr.open_dataset(area_path)["areacello"]
filenames_so = glob.glob(so_path+"*.nc")

# Mask for Arctic Ocean
ao_east = np.array([[0., 80.], [25, 80], [25, 68], [40,61], [180,66.5], [180,90], [0,90]])
ao_west = np.array([[-180,90], [-180,66.5], [-160,66.5], [-87,66.5], [-85,70], [-40,70], [-30., 80.], [0,80], [0,90]])
ao_reg = regionmask.Regions([ao_east, ao_west], names=["West", "East"], abbrevs=["AOw", "AOe"])
ao_regmask = ao_reg.mask(areacello, lon_name=lon_name, lat_name=lat_name, wrap_lon=lon360)

# Read & calculate mean AO salinity
so_avg_ao_path = glob.glob("/home/omehling/work/cmip6/piControl_salinity/so_mean_ao/soao_*{}*.nc".format(
    model_id.replace('_','*')
))[0]
so_avg_ao = xr.open_dataset(so_avg_ao_path, use_cftime=True)["so"]
so_ref = so_avg_ao.mean().item()

# Initialize slurm job
cluster = SLURMCluster(cores=16, memory="20 GB", walltime="4:00:00")
cluster.scale(jobs = 1)
client = Client(cluster)

# Create subdirectory if it does not exist already
pathlib.Path("/home/omehling/work/cmip6/piControl_salinity/fwc_ao/"+model_id).mkdir(parents=True, exist_ok=True)

# Processing
for fn in filenames_so:
    fn_out = fn.split("/")[-1].replace("so_", "fwc_") # filename for output
    
    so_sel = xr.open_dataset(fn, chunks={"time": 12}, use_cftime=True)["so"] # load so as dask array
    omask = ~so_sel.isel({"time":0, lev_name:0}, drop=True).isnull().compute() # sea-land mask
    layer_weights = layer_weights_all(so_sel[lev_name], coord=lev_name)
    
    fwcont_sel = ((so_ref - so_sel.where(so_sel < so_ref))/so_ref).weighted(layer_weights).sum(lev_name).where(omask).compute()
    fwcont_out = add_encoding(fwcont_sel, so_sel, lat_name, lon_name)
    fwcont_out.to_netcdf("/home/omehling/work/cmip6/piControl_salinity/fwc_ao/"+model_id+"/"+fn_out)
    
    print(fn)

cluster.close()