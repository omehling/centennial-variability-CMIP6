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

vertices_lat = xr.open_dataset("/home/omehling/synda-CMIP/IPSL/IPSL-CM6A-LR/piControl/r1i1p1f1/Omon/thetao/gn/v20200326/thetao_Omon_IPSL-CM6A-LR_piControl_r1i1p1f1_gn_185001-194912.nc")["bounds_nav_lat"]
vertices_lon = xr.open_dataset("/home/omehling/synda-CMIP/IPSL/IPSL-CM6A-LR/piControl/r1i1p1f1/Omon/thetao/gn/v20200326/thetao_Omon_IPSL-CM6A-LR_piControl_r1i1p1f1_gn_185001-194912.nc")["bounds_nav_lon"]
vertices_lat["x"] = range(len(vertices_lat["x"]))
vertices_lat["y"] = range(len(vertices_lat["y"]))
vertices_lon["x"] = range(len(vertices_lon["x"]))
vertices_lon["y"] = range(len(vertices_lon["y"]))
lev_bnds = xr.open_dataset("/home/omehling/synda-CMIP/IPSL/IPSL-CM6A-LR/piControl/r1i1p1f1/Omon/thetao/gn/v20200326/thetao_Omon_IPSL-CM6A-LR_piControl_r1i1p1f1_gn_185001-194912.nc")["olevel_bounds"]

root_path = "/home/omehling/synda-CMIP/IPSL/IPSL-CM6A-LR/piControl/r1i1p1f1/Omon/"
data_version = "gn/v20200326"
filenames_thetao = sorted(glob.glob(root_path+"thetao/"+data_version+"/*.nc"))
filenames_so = sorted(glob.glob(root_path+"so/"+data_version+"/*.nc"))
filenames_uo = sorted(glob.glob(root_path+"uo/"+data_version+"/*.nc"))
filenames_vo = sorted(glob.glob(root_path+"vo/"+data_version+"/*.nc"))
print(len(filenames_thetao), len(filenames_uo), len(filenames_uo), len(filenames_vo))

s_mean = xr.open_dataset("/home/omehling/work/cmip6/piControl_salinity/so_mean_ao/soao_Omon_IPSL-CM6A-LR_piControl_r1i1p1f1_gn.nc")["so"].mean().item()
print("Mean salinity: {:.2f}".format(s_mean))

# Grid points for Fram strait
fram_x = slice(269,278)
fram_y = 312

# Grid cell weights for Fram, zonal
fram_zonal_lengths = []

for vert_lat, vert_lon in zip(
        list(vertices_lat.sel(x=fram_x, y=fram_y).values),
        list(vertices_lon.sel(x=fram_x, y=fram_y).values)):
    fram_zonal_lengths.append(cell_length_zonal(vert_lat, vert_lon))
fram_zonal_lengths = xr.DataArray(fram_zonal_lengths, coords={"x": vertices_lat.sel(x=fram_x, y=fram_y).x})
print("Fram strait length:", np.sum(fram_zonal_lengths).item()/1000, "km")

# Grid cell weights for Fram, vertical
layer_weights_ece = (lev_bnds.isel(axis_nbounds=1)-lev_bnds.isel(axis_nbounds=0))

# Total: length*depth
fram_cell_weights = fram_zonal_lengths*layer_weights_ece # in m^2

# BSO indices
bso_len = 11
svalbard_idx = [282, 310]

bso_x = xr.DataArray([svalbard_idx[0]+i for i in range(1,bso_len)], dims=['locs'])
bso_y = xr.DataArray([svalbard_idx[1]-i for i in range(1,bso_len)], dims=['locs'])

# Grid cell weights for BSO
bso_lengths = []

for vert_lat, vert_lon in zip(
        list(vertices_lat.sel(x=bso_x, y=bso_y).values),
        list(vertices_lon.sel(x=bso_x, y=bso_y).values)):
    bso_lengths.append(cell_length_diagonal(vert_lat, vert_lon))
bso_lengths = xr.DataArray(bso_lengths, dims=["locs"]) # coords={"locs": vertices_lat.sel(x=bso_x, y=bso_y).locs}
print("BSO length:", np.sum(bso_lengths).item()/1000, "km")

# Total: length*depth
bso_cell_weights = bso_lengths*layer_weights_ece # in m^2

# Grid points for Davis strait
davis_x = slice(234,241)
davis_y = 290

# Grid cell weights for Davis, zonal
davis_zonal_lengths = []

for vert_lat, vert_lon in zip(
        list(vertices_lat.sel(x=davis_x, y=davis_y).values),
        list(vertices_lon.sel(x=davis_x, y=davis_y).values)):
    davis_zonal_lengths.append(cell_length_zonal(vert_lat, vert_lon))
davis_zonal_lengths = xr.DataArray(davis_zonal_lengths, coords={"x": vertices_lat.sel(x=davis_x, y=davis_y).x})
print("Davis strait length:", np.sum(davis_zonal_lengths).item()/1000, "km")

# Total: length*depth
davis_cell_weights = davis_zonal_lengths*layer_weights_ece # in m^2


# Grid points for Bering strait
bering_x = slice(114,116)
bering_y = 286

# Grid cell weights for bering, zonal
bering_zonal_lengths = []

for vert_lat, vert_lon in zip(
        list(vertices_lat.sel(x=bering_x, y=bering_y).values),
        list(vertices_lon.sel(x=bering_x, y=bering_y).values)):
    bering_zonal_lengths.append(cell_length_zonal(vert_lat, vert_lon))
bering_zonal_lengths = xr.DataArray(bering_zonal_lengths, coords={"x": vertices_lat.sel(x=bering_x, y=bering_y).x})
print("Bering strait length:", np.sum(bering_zonal_lengths).item()/1000, "km")

# Total: length*depth
bering_cell_weights = bering_zonal_lengths*layer_weights_ece # in m^2


### MAIN ###
t1 = time.time()
for load_file_idx in range(0,1): #range(len(filenames_vo)):
    ## Load data
    load_range_x = slice(110,300)
    load_range_y = slice(280,320)
    vo_ece = xr.open_dataset(filenames_vo[load_file_idx], use_cftime=True)["vo"].isel(x=load_range_x, y=load_range_y)
    vo_ece["x"] = range(load_range_x.start+1, load_range_x.stop+1)
    vo_ece["y"] = range(load_range_y.start+1, load_range_y.stop+1)
    uo_ece = xr.open_dataset(filenames_uo[load_file_idx], use_cftime=True)["uo"].isel(x=load_range_x, y=load_range_y)
    uo_ece["x"] = range(load_range_x.start+1, load_range_x.stop+1)
    uo_ece["y"] = range(load_range_y.start+1, load_range_y.stop+1)
    s_ece = xr.open_dataset(filenames_so[load_file_idx], use_cftime=True)["so"].isel(x=load_range_x, y=load_range_y)
    s_ece["x"] = range(load_range_x.start+1, load_range_x.stop+1)
    s_ece["y"] = range(load_range_y.start+1, load_range_y.stop+1)
    s_ref = s_mean

    ## Fram strait
    s_fram = s_ece.sel(x=fram_x, y=fram_y)
    vmid_fram = (vo_ece.sel(x=fram_x, y=fram_y)+vo_ece.sel(x=fram_x, y=fram_y-1))/2
    fram_fwt_monthly = (vmid_fram*((s_ref-s_fram)/s_ref)*fram_cell_weights).sum(["olevel","x"])

    ## BSO
    vo_bso_prov = vo_ece.sel(x=bso_x, y=bso_y)
    uo_bso_prov = uo_ece.sel(x=bso_x, y=bso_y)
    vel_bso_prov = uo_bso_prov/np.sqrt(2)+vo_bso_prov/np.sqrt(2) # projected to normal vector 1/sqrt(2)*(1,1)
    s_bso = s_ece.sel(x=bso_x, y=bso_y)
    bso_fwt_monthly = (vel_bso_prov*((s_ref-s_bso)/s_ref)*bso_cell_weights).sum(["olevel","locs"])

    ## Davis strait
    s_davis = s_ece.sel(x=davis_x, y=davis_y)
    vmid_davis = (vo_ece.sel(x=davis_x, y=davis_y)+vo_ece.sel(x=davis_x, y=davis_y-1))/2
    davis_fwt_monthly = (vmid_davis*((s_ref-s_davis)/s_ref)*davis_cell_weights).sum(["olevel","x"])

    ## Bering strait
    s_bering = s_ece.sel(x=bering_x, y=bering_y)
    vmid_bering = (vo_ece.sel(x=bering_x, y=bering_y)+vo_ece.sel(x=bering_x, y=bering_y-1))/2
    bering_fwt_monthly = (vmid_bering*((s_ref-s_bering)/s_ref)*bering_cell_weights).sum(["olevel","x"])

    ## Write to CSV
    df_fwts = pd.concat([fram_fwt_monthly.to_pandas(), bso_fwt_monthly.to_pandas(), davis_fwt_monthly.to_pandas(), bering_fwt_monthly.to_pandas()], axis=1)*m3s_km3yr
    df_fwts.columns = ["fwt_fram", "fwt_bso", "fwt_davis", "fwt_bering"]

    df_fwts.to_csv("fwt_ipsl/fwt_IPSL_{}.csv".format(load_file_idx))
    print("Finished {}".format(filenames_so[load_file_idx].split("/")[-1]))

t2 = time.time()
print("Script finished, total {:.3f} hrs".format((t2-t1)/60/60))
