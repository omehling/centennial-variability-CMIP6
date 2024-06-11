#!/usr/bin/env python3
import xarray as xr
import numpy as np
import pandas as pd
import scipy.fft as fft
from scipy.stats import norm as gaussian
from scipy.signal import correlate as ccorrelate

def lagregs(ds_a, ds_b, shifts, lrdim='time'):
    """Lagged regression between two Xarray datasets.
    
    Args:
        ds_a (xr.DataArray): Regressor (1D)
        ds_b (xr.DataArray): Regressand (1D+)
        shift (list or np.array): List of lags to compute (typically an np.arange)
        lrdim (str, optional): Name of the time dimension (default: "time")
    
    Note that the time dimension needs to have numerical (NOT datetime) values
    
    Returns:
        xr.DataArray: Lagged regression coefficients
    """
    lagreg_fields=[]

    for shift in shifts:
        a_aligned, b_aligned = xr.align(ds_a,ds_b.assign_coords({lrdim: ds_b[lrdim].values-shift}), join="inner")
        #print(shift,len(a_aligned))
        a_demeaned = a_aligned - a_aligned.mean(lrdim)
        lagreg_ab=(a_demeaned*b_aligned).sum(dim=lrdim)/(a_demeaned**2).sum(dim=lrdim)
        lagreg_fields.append(lagreg_ab)
        
    return xr.concat(lagreg_fields,pd.Index(shifts, name="lag"))

def linreg_np(x, y):
    """Regress each data point of gridded time series y on time series x
    
    Args:
        x (np.array): Regressor (1D)
        y (np.array): Regressand (1D+). First dimension must be identical to that of x
    
    Returns:
        np.array: Lagged regression coefficients
    """
    x_dims = len(x.shape)
    y_dims = len(y.shape)
    if len(x) != y.shape[0]:
        raise ValueError("x and y must have the same first axis")
    if x_dims != 1:
        raise ValueError("x must have exactly one dimension")
    # Linreg equation: sum (x_i - x_mean)(y_i - y_mean) / sum (x_i-x_mean)^2
    x_nom = (x-np.mean(x))
    lr_denom = np.sum(x_nom**2)
    if y_dims > 1:
        axis_expand = tuple(range(1,y_dims))
        x_nom = np.expand_dims(x_nom, axis=axis_expand)
    lr_nom = np.sum(x_nom*(y-np.mean(y, axis=0)), axis=0)
    
    return lr_nom/lr_denom

def ebisuzaki_surrogate(ts,rng=None):
    """Construct surrogate time series based on algorithm of
    Ebisuzaki, J Clim, 1997, https://doi.org/10.1175/1520-0442(1997)010<2147:AMTETS>2.0.CO;2

    Args:
        ts (np.array): Time series (1D)
        rng (Generator, optional): Random number generator
                                   (default: None -> np.random.default_rng())

    Returns:
        np.array: Surrogate time series
    """

    # Initialize random number generator (if not passed via rng option)
    if rng is None:
        rng = np.random.default_rng()
    
    # Step 1: apply (real) Fourier transform on original data
    ts_fft=fft.rfft(ts-np.mean(ts))
    #ts_fftfreq=fft.rfftfreq(len(ts))
    
    # Step 2: randomly phased Fourier components
    theta_k=rng.uniform(0,2*np.pi,size=len(ts_fft))
    rnd_r=np.abs(ts_fft)*np.exp(1.j*theta_k) # r tilde from the paper
    rnd_r[0]=0.+0.j
    rnd_r[-1]=np.sqrt(2)*np.abs(ts_fft[-1])*np.cos(theta_k[-1])+0.j
    
    # Step 3: Inverse Fourier transform
    ts_surr=fft.irfft(rnd_r) # surrogate time series
    
    return ts_surr

def confidence_levels_lagreg(ts1, ts2, lag_range, N_surr=1000, ci_level=0.9, time_dim="time"):
    """Confidence intervals for lagged regression based on
    Ebisuzaki (1997) surrogate time series (see `ebisuzaki_surrogate`).
    This function accounts for multiple comparisons using effective
    degrees of freedom based on the autocorrelation of both time series.

    Args:
        ts1 (np.array or xr.DataArray): Regressor (1D)
        ts1 (np.array or xr.DataArray): Regressand (1D)
        lag_range (list or np.array): List of lags as in `lagregs`
        N_surr (int, optional): number of surrogates (default: 1000)
        ci_level (float, optional): Confidence level (default: 0.9)
        time_dim (str, optinal): Name of the time dimension (default: "time")

    Returns:
        lower_conf, upper_conf (float): Confidence levels
    """
    
    # Convert numpy to xarray objects
    if "xarray" not in str(type(ts1)):
        ts1 = xr.DataArray(ts1, coords={time_dim: np.arange(1,len(ts1)+1)})
    if "xarray" not in str(type(ts2)):
        ts1 = xr.DataArray(ts2, coords={time_dim: np.arange(1,len(ts2)+1)})
    time_axis = ts1[time_dim]
    t_len = len(ts1[time_dim])
    t2_len = len(ts2[time_dim])
        
    # Sanity checks
    if t_len != t2_len:
        raise ValueError("both time series must have the same length for lagged regression but have lengths {} and {}".format(t_len, t2_len))
    if t_len%2 == 1:
        print("Warning: confidence_level_lagreg can only handle time series with an even number of data points. Truncating last value")
        ts1 = ts1[:-1]
        ts2 = ts2[:-1]
        time_axis = ts1[time_dim]
        t_len = len(ts1[time_dim])
    
    # Surrogates using the Ebisuzaki method
    surrs = np.zeros((N_surr, t_len))
    for b in range(N_surr):
        surrs[b,:] = ebisuzaki_surrogate(ts1.values)[:]
    
    # Instantaneous regression for each surrogate
    reg_surrs = lagregs(
        xr.DataArray(surrs, coords={"sample": range(N_surr), time_dim: time_axis}),
        ts2, [0], time_dim
    )
    
    # Effective degrees of freedom (formula by von Storch & Zwiers 1999, also Mudelsee 2014 eq. 2.37)
    acf_ts1 = (ccorrelate(ts1, ts1)/ccorrelate(ts1, ts1).max())[t_len-1:]
    acf_ts2 = (ccorrelate(ts2, ts2)/ccorrelate(ts2, ts2).max())[t_len-1:]
    eff_deg_denom = 1+2*np.sum(((1-np.arange(len(acf_ts1))/len(acf_ts1))*acf_ts1*acf_ts2)[:len(lag_range)])
    
    sample_norm_params = gaussian.fit(reg_surrs.squeeze().values)
    lower_conf, upper_conf = gaussian.ppf([(1-ci_level)/(len(lag_range)/eff_deg_denom), 1-(1-ci_level)/(len(lag_range)/eff_deg_denom)], *sample_norm_params) # Bonferoni correction
    
    return lower_conf, upper_conf