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

aux_path = "/home/omehling/synda-CMIP/NCAR/CESM2/piControl/r1i1p1f1/Omon/thetao/gn/v20190320/thetao_Omon_CESM2_piControl_r1i1p1f1_gn_000101-009912.nc"
vertices_lat = xr.open_dataset(aux_path)["lat_bnds"].rename({"nlon": "i", "nlat": "j"})
vertices_lon = xr.open_dataset(aux_path)["lon_bnds"].rename({"nlon": "i", "nlat": "j"})
lev_bnds = xr.open_dataset(aux_path)["lev_bnds"]
# Vertical layer weights
layer_weights_ece = lev_bnds.isel(d2=1)-lev_bnds.isel(d2=0)

root_path = "/home/omehling/synda-CMIP/NCAR/CESM2/piControl/r1i1p1f1/Omon/"
filenames_thetao = sorted(glob.glob(root_path+"thetao/"+"gn/*/*.nc"))
filenames_so = sorted(glob.glob(root_path+"so/"+"gn/*/*.nc"))
filenames_uo = sorted(glob.glob(root_path+"uo/"+"gn/*/*.nc"))
filenames_vo = sorted(glob.glob(root_path+"vo/"+"gn/*/*.nc"))
print(len(filenames_thetao), len(filenames_so), len(filenames_uo), len(filenames_vo))

s_mean = xr.open_dataset("/home/omehling/work/cmip6/piControl_salinity/so_mean_ao/soao_Omon_CESM2_piControl_r1i1p1f1_gr.nc")["so"].mean().item()
print("Mean salinity: {:.2f}".format(s_mean))

# Fram strait indices
fram_left_idx = [98,370]
fram_right_idx = [106,378]

fram_i = xr.DataArray([fram_left_idx[0]+i for i in range(fram_right_idx[0]-fram_left_idx[0]+1)], dims=['locs'])
fram_j = xr.DataArray([fram_left_idx[1]+i for i in range(fram_right_idx[0]-fram_left_idx[0]+1)], dims=['locs'])

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
bso_left_idx = [79,355]
bso_right_idx = [90,366]

bso_i = xr.DataArray([bso_left_idx[0]+i for i in range(bso_right_idx[0]-bso_left_idx[0]+1)], dims=['locs'])
bso_j = xr.DataArray([bso_left_idx[1]+i for i in range(bso_right_idx[0]-bso_left_idx[0]+1)], dims=['locs'])

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
davis_i = slice(294,305)
davis_i_m1 = slice(294-1,305-1)
davis_j=366

# Grid cell weights for Davis, zonal
davis_zonal_lengths = []

for vert_lat, vert_lon in zip(
        list(vertices_lat.sel(i=davis_i, j=davis_j).values),
        list(vertices_lon.sel(i=davis_i, j=davis_j).values)):
    davis_zonal_lengths.append(cell_length_zonal(vert_lat, vert_lon))
davis_zonal_lengths = xr.DataArray(davis_zonal_lengths, coords={"i": vertices_lat.sel(i=davis_i, j=davis_j).i})
print("Davis strait length:", np.sum(davis_zonal_lengths).item()/1000, "km")

# Total: length*depth
davis_cell_weights = davis_zonal_lengths*layer_weights_ece # in m^2

# Bering strait indices
bering_i = slice(200,201)
bering_j = 333
bering_coords_u1 = {"i": 200, "j": 333} # u1 (vo north of strait)
bering_coords_u0 = {"i": 200, "j": 332} # u0 (vo south of strait)

# Grid cell weights for Bering, zonal
bering_zonal_lengths = []

for vert_lat, vert_lon in zip(
        list(vertices_lat.sel(i=bering_i, j=bering_j).values),
        list(vertices_lon.sel(i=bering_i, j=bering_j).values)):
    bering_zonal_lengths.append(cell_length_zonal(vert_lat, vert_lon))
bering_zonal_lengths = xr.DataArray(bering_zonal_lengths, coords={"i": vertices_lat.sel(i=bering_i, j=bering_j).i})
print("Bering strait length:", np.sum(bering_zonal_lengths).item()/1000, "km")

# Total: length*depth
bering_cell_weights = bering_zonal_lengths*layer_weights_ece # in m^2


### MAIN ###
t1 = time.time()
for load_file_idx in range(len(filenames_vo)):
    ## Load data
    vo_ece = xr.open_dataset(filenames_vo[load_file_idx], use_cftime=True)["vo"].rename({"nlon": "i", "nlat": "j"})
    uo_ece = xr.open_dataset(filenames_uo[load_file_idx], use_cftime=True)["uo"].rename({"nlon": "i", "nlat": "j"})
    s_ece = xr.open_dataset(filenames_so[load_file_idx], use_cftime=True)["so"].rename({"nlon": "i", "nlat": "j"})
    s_ref = s_mean

    # (u,v) are defined on the top right corner of each grid cell,
    # so midpoint (u,v) at tracer locations are given by (u_i,j+u_i-1,j-1)/2
    # u goes into the Arctic, v out of the Arctic, so v must be multiplied by -1

    ## Fram strait
    s_fram = s_ece.sel(i=fram_i, j=fram_j)
    vo_mid_fram = (vo_ece.sel(i=fram_i, j=fram_j)+vo_ece.sel(i=fram_i-1, j=fram_j-1))/2
    uo_mid_fram = (uo_ece.sel(i=fram_i, j=fram_j)+uo_ece.sel(i=fram_i-1, j=fram_j-1))/2
    vel_fram = uo_mid_fram/np.sqrt(2)-vo_mid_fram/np.sqrt(2) # projected to normal vector 1/sqrt(2)*(1,-1)

    fram_fwt_monthly = (vel_fram*((s_ref-s_fram)/s_ref)*fram_cell_weights).sum(["lev","locs"])

    ## BSO
    s_bso = s_ece.sel(i=bso_i, j=bso_j)
    vo_mid_bso = (vo_ece.sel(i=bso_i, j=bso_j)+vo_ece.sel(i=bso_i-1, j=bso_j-1))/2
    uo_mid_bso = (uo_ece.sel(i=bso_i, j=bso_j)+uo_ece.sel(i=bso_i-1, j=bso_j-1))/2
    vel_bso = uo_mid_bso/np.sqrt(2)-vo_mid_bso/np.sqrt(2) # projected to normal vector 1/sqrt(2)*(1,-1)

    bso_fwt_monthly = (vel_bso*((s_ref-s_bso)/s_ref)*bso_cell_weights).sum(["lev","locs"])

    ## Davis strait
    s_davis = s_ece.sel(i=davis_i, j=davis_j)
    vo_topright = vo_ece.sel(i=davis_i, j=davis_j)
    vo_bottomleft = vo_ece.sel(i=davis_i_m1, j=davis_j-1)
    vo_bottomleft = vo_bottomleft.reindex(i=vo_bottomleft.i+1)
    vo_davis = (vo_topright+vo_bottomleft)/2

    davis_fwt_monthly = (vo_davis*((s_ref-s_davis)/s_ref)*davis_cell_weights).sum(["lev","i"])

    ## Bering strait
    s_bering = s_ece.sel(i=bering_i, j=bering_j)
    vo_bering = (vo_ece.sel(i=bering_coords_u1["i"], j=bering_coords_u1["j"])
                +vo_ece.sel(i=bering_coords_u0["i"], j=bering_coords_u0["j"]))/4

    bering_fwt_monthly = (vo_bering*((s_ref-s_bering)/s_ref)*bering_cell_weights).sum(["lev","i"])

    ## Write to CSV
    df_fwts = pd.concat([fram_fwt_monthly.to_pandas(), bso_fwt_monthly.to_pandas(), davis_fwt_monthly.to_pandas(), bering_fwt_monthly.to_pandas()], axis=1)*m3s_km3yr
    df_fwts.columns = ["fwt_fram", "fwt_bso", "fwt_davis", "fwt_bering"]

    df_fwts.to_csv("fwt_cesm/fwt_CESM_{:02d}.csv".format(load_file_idx))
    print("Finished {}".format(filenames_so[load_file_idx].split("/")[-1]))

t2 = time.time()
print("Script finished, total {:.3f} hrs".format((t2-t1)/60/60))