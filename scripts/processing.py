#!/usr/bin/env python3
import numpy as np
import xarray as xr
from scipy.signal import convolve, butter, sosfiltfilt

## ----------
## Detrending
## ----------
def detrend_ts(ts, degree, time=None):
    """Detrend a one-dimensional time series by fitting a polynomial.
    
    Args:
        ts (np.array): Time series
        degree (int): Degree of the polynomial
        time (np.array, optional): Numerical time coordinate (if not evenly spaced)
    
    Returns:
        np.array: The detrended time series
    """
    if time is None:
        time=np.arange(len(ts))
    z=np.polyfit(time, ts, degree)
    p=np.poly1d(z)
    return ts-p(time)
    
def detrend_xr(ds, degree, keep_mean=True, time_dim="time"):
    """Detrend an xarray object by fitting a polynomial.
    
    Args:
        ds (xr.DataArray): Input data
        degree (int): Degree of the polynomial
        keep_mean (bool, optional): Keep the time mean (default: True)
        time_dim (str, optional): Name of the time dimension (default: "time")
    
    Returns:
        xr.DataArray: The detrended data
    """
    if keep_mean:
        mean = ds.mean(time_dim)
    ds_detr=xr.apply_ufunc(
        detrend_ts,
        ds,
        degree,
        input_core_dims=[[time_dim],[]],output_core_dims=[[time_dim]],
        vectorize=True
    )
    if keep_mean:
        return ds_detr+mean
    else:
        return ds_detr

## ------------------
## Low-pass filtering
## ------------------
def lanczos_weights(window, cutoff):
    """Calculate weights for a low pass Lanczos filter.

    Args:
        window (int): The length of the filter window.
        cutoff (float): The cutoff frequency in inverse time steps.
        
    References:
        from https://scitools.org.uk/iris/docs/v1.2/examples/graphics/SOI_filtering.html
    """
    order = ((window - 1) // 2 ) + 1
    nwts = 2 * order + 1
    w = np.zeros([nwts])
    n = nwts // 2
    w[n] = 2 * cutoff
    k = np.arange(1., n)
    sigma = np.sin(np.pi * k / n) * n / (np.pi * k)
    firstfactor = np.sin(2. * np.pi * cutoff * k) / (np.pi * k)
    w[n-1:0:-1] = firstfactor * sigma
    w[n+1:-1] = firstfactor * sigma
    return w[1:-1]

def filter_padding(ts, window, pad_type='mirror', detrend=True, detrend_degree=1):
    """Pad time series prior to application of a low-pass filter.
    
    Args:
        ts (np.array): Input time series
        window (int): Length of the padding window
        pad_type ("mirror" or "periodic"): Padding type (default: "mirror")
        detrend (bool, optional): Detrend time series prior to padding
            (trend will be added again before returning) (default: True)
        detrend_degree (int, optional): Degree for detrending polynomial (default: 1)
    
    Returns:
        np.array: Padded time series of length N+2*window
    """
    ts_pad = np.zeros(2*window+len(ts))
    if detrend:
        t = np.arange(len(ts))
        z = np.polyfit(t,ts,detrend_degree)
        p = np.poly1d(z)
        ts_in = ts-p(t)
    else:
        ts_in = ts
    ts_pad[window:-window] = ts_in[:]
    if pad_type == 'mirror':
        ts_pad[:window] = ts_in[:window][::-1]
        ts_pad[-window:] = ts_in[-window:][::-1]
    elif pad_type == 'periodic':
        ts_pad[:window] = ts_in[-window:]
        ts_pad[-window:] = ts_in[:window]
    else:
        raise ValueError('in filter_padding: pad_type must be one of "mirror" or "periodic".')
    if detrend:
        t_pad = np.arange(-window,len(ts)+window)
        ts_pad = ts_pad+p(t_pad)
    return ts_pad

def filter_ts(ts, cutoff, filter_type='butter', butter_order=4, **kwargs):
    """Apply low-pass filter with padding.

    Args:
        ts (np.array): Input time series
        cutoff (float): Cutoff **period** in multiples of the input sampling (e.g. years for yearly input)
        filter_type ("lanczos" or "butter"): Low-pass filter (default: butter)
        butter_order (int): Order of Butterworth filter (default: 4)
        **kwargs: Arguments to be passed to `filter_padding`
    
    Returns:
        np.array: Filtered time series
    """
    #   padding (bool, optional): Apply padding (default: True) [padding=False not yet implemented]
    weights=lanczos_weights(cutoff*2+1,1./cutoff) # weights for Lanczos filter
    n_pad=int(np.ceil(len(weights)/2))
    
    # Padding
    ts_mirr=filter_padding(ts,n_pad,**kwargs)
    
    # Filtering
    if filter_type=='lanczos':
        # Lanczos filter
        return convolve(ts_mirr,weights,'same')[n_pad:-n_pad]
    elif filter_type=='butter':
        # 4th-order Butterworth filter
        sos = butter(butter_order, 1./cutoff, 'lowpass', fs=1, output='sos')
        return sosfiltfilt(sos, ts_mirr-np.mean(ts_mirr))[n_pad:-n_pad]+np.mean(ts_mirr)
    else:
        raise ValueError('in filter_ts: filter_type must be one of "lanczos" or "butter".')

def filter_xr(ds, cutoff, time_dim="time", **kwargs):
    """Xarray wrapper for `filter_ts`.
    
    Args:
        ds (xr.DataArray): Input data array
        cutoff (float): Cutoff **period** in multiples of the input sampling (e.g. years for yearly input)
        time_dim (string): Name of time dimension (default: "time")
        **kwargs: Arguments to be passed to `filter_ts` and `filter_padding`
    
    Returns:
        xr.DataArray: Filtered data array
    """
    return xr.apply_ufunc(
        filter_ts,
        ds,
        cutoff,
        kwargs=kwargs,
        input_core_dims=[[time_dim],[]],
        output_core_dims=[[time_dim]],
        vectorize=True
    )

## -------------
## Miscellaneous
## -------------
def lon_to_180(lon):
    """Transform longitude range from (0,360) to (-180, 180)"""
    lv=lon.copy()
    lvn=np.zeros(len(lv))
    for i in range(len(lv)):
        if lv[i]>180:
            lvn[i]=lv[i]-360
        else:
            lvn[i]=lv[i]
    return lvn

def ds_to_180(ds):
    """Transform longitude range of xarray data set from (0,360) to (-180, 180)"""
    dsr=ds.copy()
    dsr['lon']=lon_to_180(dsr.lon)
    return dsr.sortby('lon')

def lat_lon_name(var):
    """Return name of latitude and longitude coordinates from input dataset"""
    if "lat" in list(var.coords):
        lat_name = "lat"
        lon_name = "lon"
    elif "latitude" in list(var.coords):
        lat_name = "latitude"
        lon_name = "longitude"
    if "nav_lat" in list(var.coords):
        lat_name = "nav_lat"
        lon_name = "nav_lon"
    return lat_name, lon_name

def time2year(ds):
    """Convert 'time' axis into 'year' axis on dataset that already contains annual means"""
    if "year" in list(ds.coords):
        return ds.swap_dims({"time": "year"}).drop("time")
    elif type(ds["time"].values[0]) == np.int64 or type(ds["time"].values[0]) == np.int32:
        return ds.rename({"time": "year"})
    else:
        ds["time"] = [t.year for t in ds["time"].values]
        return ds.rename({"time": "year"})