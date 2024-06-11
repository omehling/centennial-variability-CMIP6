#!/usr/bin/env python3

import numpy as np
import nitime.algorithms as tsa
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import chi2

## Helper functions
def s_spec(f,S0,r,fN=0.5):
    '''
    Analytical AR(1) spectrum
    Args:
        f (array): Frequencies
        S0 (float): Variance
        r ([-1, 1]): Persistence
        fN: Nyquist frequency (default: 0.5)
    Returns: Array of len(f)
    '''
    return S0*(1-r**2)/(1-2*r*np.cos(np.pi*f/fN)+r**2)

def s_log(f,S0,r,fN=0.5):
    '''
    Logarithm of analytical AR(1) spectrum
    Args:
        f (array): Frequencies
        S0 (float): Variance
        r ([-1, 1]): Persistence
        fN: Nyquist frequency (default: 0.5)
    Returns: Array of len(f)
    '''
    return np.log(s_spec(f,S0,r,fN))

# Median smoothing following Mann & Lees (1996)
def median_smoothing(freqs,psd,dfsmooth=0.075):
    '''
    Returns median-smoothed spectrum
    Args:
        freqs (array): Frequencies
        psd (array): Unsmoothed multi-taper power spectrum
        dfsmooth: Width of the smoothing window
                  in frequency units (default: 0.075)
        fN: Nyquist frequency (default: 0.5)
    Returns:
        smoothed_psd: Smoothed spectrum with same length as psd
    '''
    smoothed_psd=np.zeros(len(psd))
    for i, fr in enumerate(freqs):
        window_left=max(fr-dfsmooth,np.min(freqs))
        window_right=min(fr+dfsmooth,np.max(freqs))
        med=np.median(psd[np.where(np.logical_and(freqs>=window_left,freqs<=window_right))])
        smoothed_psd[i]=med
    return smoothed_psd

# Compute confidence bounds from chi2 distribution
def chi2conf(K, Sxx=1, ci=.99):
    '''
    Returns confidence bounds computed using chi-square 
    distribution.
    Args: 
        K (int): Number of tapers
        Sxx (array): Multitaper spectrum (optional)
        ci ([0.5, 1]): Confidence level (default: 0.99)
    Returns:
        lb, ub: lower and upper bounds
    '''
    ub = 2 * K / chi2.ppf(1 - ci, 2 * K) * Sxx
    lb = 2 * K / chi2.ppf(    ci, 2 * K) * Sxx
    return lb, ub

## 
class MTMSpectrum:
    def __init__(self, ts, sampling_interval=1, suppress_warnings=False):
        self.ts = ts
        self.sampling = sampling_interval
        self.suppress_warnings = suppress_warnings
        self.freqs = None
        self.psd = None
        self.Kmax = 0
        self.s0_opt = 0
        self.r_opt = 0
        self.smoothed = None
        self.ar1_fit = None
        self.estimate()
    def estimate(self, adaptive=False, jackknife=False, **kwargs):
        freqs, psd, nu = tsa.multi_taper_psd(self.ts, Fs=1./self.sampling,
                                             adaptive=adaptive, jackknife=jackknife, **kwargs)
        self.freqs = freqs
        self.psd = psd
        self.Kmax = nu[0]/2
    def fit(self,dfsmooth=0.075,fit_mode='normal',freq_range=None):
        if freq_range is None:
            smooth_freqs = self.freqs
            smooth_psd_in = self.psd
        else:
            smooth_freqs = self.freqs[np.where(np.logical_and(self.freqs>=freq_range[0],self.freqs<=freq_range[1]))]
            smooth_psd_in = self.psd[np.where(np.logical_and(self.freqs>=freq_range[0],self.freqs<=freq_range[1]))]
        # Apply median smoothing
        smoothed = median_smoothing(
            #self.freqs, self.psd, dfsmooth=dfsmooth#, fN=1/(2*self.sampling)
            smooth_freqs, smooth_psd_in, dfsmooth=dfsmooth
        )
        self.smoothed = smoothed
        self._smooth_freqs = smooth_freqs
        # Fit parameters analytical AR(1) spectrum
        if fit_mode == 'log':
            (s0_opt, r_opt),_=curve_fit(s_log, smooth_freqs, np.log(smoothed), p0=[smoothed.mean(),0.5])
        else:
            (s0_opt, r_opt),_=curve_fit(s_spec, smooth_freqs, smoothed, p0=[smoothed.mean(),0.5])
        self.s0_opt = s0_opt
        self.r_opt = r_opt
        # Compute fit
        ar1_fit = s_spec(self.freqs,s0_opt,r_opt)
        self.ar1_fit = ar1_fit
    def get_ar1fit(self):
        if self.freqs is None:
            self.estimate()
        if self.ar1_fit is None:
            if not self.suppress_warnings:
                print('Warning: Running get_ar1fit without previously running fit() - Fit using default parameters.')
            self.fit()
        return self.ar1_fit
    def get_periods(self):
        return 1./self.freqs
    def getCIs(self,ci=.99):
        lower_ci = chi2conf(self.Kmax,ci=ci)[0]*self.ar1_fit
        upper_ci = chi2conf(self.Kmax,ci=ci)[1]*self.ar1_fit
        return lower_ci, upper_ci
    ## Plot functions
    def plot_spectrum(self,x='freq',ax=None,**kwargs):
        if ax is None:
            ax=plt
        if x=='freq':
            ax.loglog(self.freqs, self.psd, **kwargs)
        elif x=='period':
            ax.loglog(1./self.freqs[1:], self.psd[1:], **kwargs)
        else:
            raise ValueError('x must be one of ("freq", "period").')
    def plot_fit(self,x='freq',ax=None,**kwargs):
        if ax is None:
            ax=plt
        if x=='freq':
            ax.loglog(self.freqs, self.ar1_fit, **kwargs)
        elif x=='period':
            ax.loglog(1./self.freqs[1:], self.ar1_fit[1:], **kwargs)
        else:
            raise ValueError('x must be one of ("freq", "period").')
    def plot_smoothed(self,x='freq',ax=None,**kwargs):
        if ax is None:
            ax=plt
        if x=='freq':
            ax.loglog(self._smooth_freqs, self.smoothed, **kwargs)
        elif x=='period':
            if self._smooth_freqs[0]==0.:
                ax.loglog(1./self._smooth_freqs[1:], self.smoothed[1:], **kwargs)
            else:
                ax.loglog(1./self._smooth_freqs, self.smoothed, **kwargs)
        else:
            raise ValueError('x must be one of ("freq", "period").')
    def plot_CIs(self,ci=.99,ci_type='both',x='freq',ax=None,label=None,**kwargs):
        lower_ci, upper_ci = self.getCIs(ci)
        if ax is None:
            ax=plt
        if x=='freq':
            if ci_type!='upper': ax.loglog(self.freqs, lower_ci,
                                           label=(label if ci_type=='lower' else None),
                                           **kwargs)
            if ci_type!='lower': ax.loglog(self.freqs, upper_ci, label=label, **kwargs)
        elif x=='period':
            if ci_type!='upper': ax.loglog(1./self.freqs[1:], lower_ci[1:],
                                           label=(label if ci_type=='lower' else None),
                                           **kwargs)
            if ci_type!='lower': ax.loglog(1./self.freqs[1:], upper_ci[1:], label=label, **kwargs)
        else:
            raise ValueError('x must be one of ("freq", "period").')