#!/usr/bin/env python3

import numpy as np
import xarray as xr
import pandas as pd
import dask
import glob
from haversine import haversine
import time

def cell_length_zonal(vertices_lat, vertices_lon):
    middle_left_coords = ((vertices_lat[3]+vertices_lat[0])/2, (vertices_lon[3]+vertices_lon[0])/2)
    middle_right_coords = ((vertices_lat[2]+vertices_lat[1])/2, (vertices_lon[2]+vertices_lon[1])/2)
    
    return haversine(middle_left_coords, middle_right_coords, normalize=True)*1000   # distance between vortices in meters

def cell_length_diagonal(vertices_lat, vertices_lon):
    upper_left_coords = (vertices_lat[3], vertices_lon[3])
    lower_right_coords = (vertices_lat[1], vertices_lon[1])
    
    return haversine(upper_left_coords, lower_right_coords, normalize=True)*1000   # distance between vortices in meters

### SETUP ###
rho_ref = 1027 # sea water density [kg/m^3]
cp_ref = 3985 # specific heat capacity [J/kg/K]
m3s_km3yr = 1e-9*86400*365.24 # conversion factor m^3/s -> km^3/yr

aux_path = "/data/datasets/synda/data/CMIP6/CMIP/HAMMOZ-Consortium/MPI-ESM-1-2-HAM/piControl/r1i1p1f1/Omon/thetao/gn/v20200120/thetao_Omon_MPI-ESM-1-2-HAM_piControl_r1i1p1f1_gn_185001-186912.nc"
vertices_lat = xr.open_dataset(aux_path)["vertices_latitude"]
vertices_lon = xr.open_dataset(aux_path)["vertices_longitude"]
lev_bnds = xr.open_dataset(aux_path)["lev_bnds"]
# Vertical layer weights
layer_weights_ece = lev_bnds.isel(bnds=1)-lev_bnds.isel(bnds=0)

root_path = "/data/datasets/synda/data/CMIP6/CMIP/HAMMOZ-Consortium/MPI-ESM-1-2-HAM/piControl/r1i1p1f1/Omon/"
filenames_thetao = sorted(glob.glob(root_path+"thetao/gn/*/*.nc"))
filenames_so = sorted(glob.glob(root_path+"so/gn/*/*.nc"))
filenames_uo = sorted(glob.glob(root_path+"uo/gn/*/*.nc"))
filenames_vo = sorted(glob.glob(root_path+"vo/gn/*/*.nc"))
print(len(filenames_thetao), len(filenames_so), len(filenames_uo), len(filenames_vo))

s_mean = xr.open_dataset("/home/omehling/work/cmip6/piControl_salinity/so_mean_ao/soao_Omon_MPI-ESM-1-2-HAM_piControl_r1i1p1f1_gn.nc")["so"].mean().item()

# Fram strait indices
fram_left_idx = [224, 35]
fram_right_idx = [216, 43]

fram_i = xr.DataArray([fram_left_idx[0]-i for i in range(-fram_right_idx[0]+fram_left_idx[0]+1)], dims=['locs'])
fram_j = xr.DataArray([fram_left_idx[1]+i for i in range(-fram_right_idx[0]+fram_left_idx[0]+1)], dims=['locs'])

# Grid cell weights for Fram strait
fram_lengths = []

for vert_lat, vert_lon in zip(
        list(vertices_lat.sel(i=fram_i, j=fram_j).values),
        list(vertices_lon.sel(i=fram_i, j=fram_j).values)):
    fram_lengths.append(cell_length_diagonal(vert_lat, vert_lon))
fram_lengths = xr.DataArray(fram_lengths, coords={"locs": vertices_lat.sel(i=fram_i, j=fram_j).locs})
print("Fram length:", np.sum(fram_lengths).item()/1000, "km")

# Total: length*depth
fram_cell_weights = fram_lengths*layer_weights_ece # in m^2

# BSO indices
bso_left_idx = [206, 48]
bso_right_idx = [196, 58]

bso_i = xr.DataArray([bso_left_idx[0]-i for i in range(-bso_right_idx[0]+bso_left_idx[0]+1)], dims=['locs'])
bso_j = xr.DataArray([bso_left_idx[1]+i for i in range(-bso_right_idx[0]+bso_left_idx[0]+1)], dims=['locs'])

# Grid cell weights for BSO
bso_lengths = []

for vert_lat, vert_lon in zip(
        list(vertices_lat.sel(i=bso_i, j=bso_j).values),
        list(vertices_lon.sel(i=bso_i, j=bso_j).values)):
    bso_lengths.append(cell_length_diagonal(vert_lat, vert_lon))
bso_lengths = xr.DataArray(bso_lengths, coords={"locs": vertices_lat.sel(i=bso_i, j=bso_j).locs})
print("BSO length:", np.sum(bso_lengths).item()/1000, "km")

# Total: length*depth
bso_cell_weights = bso_lengths*layer_weights_ece # in m^2

# Davis strait indices
davis_i = slice(67,88)
davis_j = 28

# Grid cell weights for Davis
davis_lengths = []

for vert_lat, vert_lon in zip(
        list(vertices_lat.sel(i=davis_i, j=davis_j).values),
        list(vertices_lon.sel(i=davis_i, j=davis_j).values)):
    davis_lengths.append(cell_length_zonal(vert_lat, vert_lon))
davis_lengths = xr.DataArray(davis_lengths, coords={"i": vertices_lat.sel(i=davis_i, j=davis_j).i})
print("Davis length:", np.sum(davis_lengths).item()/1000, "km")

# Total: length*depth
davis_cell_weights = davis_lengths*layer_weights_ece # in m^2

# Bering strait indices
bering_i = [7]
bering_j = 84

# Grid cell weights for Bering
bering_lengths = []

for vert_lat, vert_lon in zip(
        list(vertices_lat.sel(i=bering_i, j=bering_j).values),
        list(vertices_lon.sel(i=bering_i, j=bering_j).values)):
    bering_lengths.append(cell_length_zonal(vert_lat, vert_lon))
bering_lengths = xr.DataArray(bering_lengths, coords={"i": vertices_lat.sel(i=bering_i, j=bering_j).i})
print("Bering length:", np.sum(bering_lengths).item()/1000, "km")

# Total: length*depth
bering_cell_weights = bering_lengths*layer_weights_ece # in m^2


### MAIN ###
t1 = time.time()
for load_file_idx in range(len(filenames_vo)):
    ## Load data
    vo_ece = xr.open_dataset(filenames_vo[load_file_idx], use_cftime=True)["vo"]
    uo_ece = xr.open_dataset(filenames_uo[load_file_idx], use_cftime=True)["uo"]
    t_ece = xr.open_dataset(filenames_thetao[load_file_idx], use_cftime=True)["thetao"]
    s_ece = xr.open_dataset(filenames_so[load_file_idx], use_cftime=True)["so"]

	# v is north of tracer points for a given index, u is east of tracer points
    # u goes into the Arctic, v out of the Arctic, so v must be multiplied by -1

    s_ref = s_mean
    ## BSO
    s_bso = s_ece.sel(i=bso_i, j=bso_j)
    vo_bso_prov = vo_ece.sel(i=bso_i, j=bso_j).fillna(0)
    uo_bso_prov = uo_ece.sel(i=bso_i, j=bso_j).fillna(0)
    vel_bso_prov = uo_bso_prov/np.sqrt(2)-vo_bso_prov/np.sqrt(2) # projected to normal vector 1/sqrt(2)*(1,-1)

    bso_fwt_monthly = (vel_bso_prov*((s_ref-s_bso)/s_ref)*bso_cell_weights).sum(["lev","locs"])

    ## Fram
    s_fram = s_ece.sel(i=fram_i, j=fram_j)
    vo_fram_prov = vo_ece.sel(i=fram_i, j=fram_j).fillna(0)
    uo_fram_prov = uo_ece.sel(i=fram_i, j=fram_j).fillna(0)
    vel_fram_prov = uo_fram_prov/np.sqrt(2)-vo_fram_prov/np.sqrt(2) # projected to normal vector 1/sqrt(2)*(1,-1)

    fram_fwt_monthly = (vel_fram_prov*((s_ref-s_fram)/s_ref)*fram_cell_weights).sum(["lev","locs"])

    ## Davis
    s_davis = s_ece.sel(i=davis_i, j=davis_j)
    vo_davis = (vo_ece.sel(i=davis_i, j=davis_j).fillna(0)+vo_ece.sel(i=davis_i, j=davis_j+1).fillna(0))/2

    davis_fwt_monthly = (vo_davis*((s_ref-s_davis)/s_ref)*davis_cell_weights).sum(["lev","i"])

    ## Bering
    s_bering = s_ece.sel(i=bering_i, j=bering_j)
    vo_bering = (vo_ece.sel(i=bering_i, j=bering_j).fillna(0)+vo_ece.sel(i=bering_i, j=bering_j-1).fillna(0))/2

    bering_fwt_monthly = (vo_bering*((s_ref-s_bering)/s_ref)*bering_cell_weights).sum(["lev","i"])

    ## Write to CSV
    df_fwts = pd.concat([fram_fwt_monthly.to_pandas(), bso_fwt_monthly.to_pandas(), davis_fwt_monthly.to_pandas(), bering_fwt_monthly.to_pandas()], axis=1)*m3s_km3yr
    df_fwts.columns = ["fwt_fram", "fwt_bso", "fwt_davis", "fwt_bering"]

    df_fwts.to_csv("fwt_mpi-ham/fwt_MPI-HAM_{:02d}.csv".format(load_file_idx))
    print("Finished {}".format(filenames_thetao[load_file_idx].split("/")[-1]))

t2 = time.time()
print("Script finished, total {:.3f} hrs".format((t2-t1)/60/60))