"""
Test suite for the TD to TFD transform
"""

import pytest

import numpy as np
import matplotlib.pyplot as plt

import jax.numpy as jnp

import WDM
import TimeFrequencyWaveforms as TFW
from TimeFrequencyWaveforms.code import utils



def amplitude(t, par):
    tc = par['tc']
    w = par['w']
    return jnp.exp(-0.5*((t-tc)/w)**2)

def phase(t, par):
    tc = par['tc']
    f = par['f']
    fdot = par['fdot']
    phic = par['phic']
    return phic + 2.*jnp.pi*f*(t-tc) + jnp.pi*fdot*(t-tc)**2

def frequency(t, par):
    f = par['f']
    tc = par['tc']
    fdot = par['fdot']
    return f + fdot * (t-tc)

def frequency_deriv(t, par):
    fdot = par['fdot']
    return fdot * jnp.ones_like(t) 

def wavelet_inner_product(wdm : WDM.WDM.WDM_transform,
                          Anm : jnp.ndarray, Bnm : jnp.ndarray, 
                          f_low : float=None, f_high : float=None, 
                          t_low : float=None, t_high : float=None) -> float:
    """ 
    Compute white TF noise inner product between two signals with wavelet
    coefficients Anm and Bnm. This is a sum over the time-frequency grid.
    The optional arguments allow this sum to be restricted to a sub-region
    of the time-frequency grid.

    Parameters
    ---------- 
    wdm : WDM.WDM.WDM_transform
        An instance of the WDM wavelet transform class.
    Anm : jnp.ndarray
        Wavelet coefficients of first signal, shape=(Nt, Nf).
    Bnm : jnp.ndarray
        Wavelet coefficients of second signal, shape=(Nt, Nf).
    f_low : float
        Lower frequency bound of inner product. Optional.
    f_high : float
        Upper frequency bound of inner product. Optional.
    t_low : float
        Lower time bound of inner product. Optional.
    t_high : float
        Upper time bound of inner product. Optional.

    Returns
    -------
    AB : float
        The inner product.
    """
    tn = wdm.dT * jnp.arange(wdm.Nt)
    fm = wdm.dF * jnp.arange(wdm.Nf)

    t_low = t_low if t_low is not None else 0.0
    t_high = t_high if t_high is not None else wdm.T
    f_low = f_low if f_low is not None else 0.0
    f_high = f_high if f_high is not None else wdm.f_Ny

    mask = jnp.outer( jnp.logical_and(tn > t_low, tn < t_high),
                      jnp.logical_and(fm > f_low, fm < f_high) )

    AB = jnp.sum(Anm[mask]*Bnm[mask])

    return AB


def test_transform():
    """Test the transform for a simple, linearly chriping Gaussian wavelet."""
    dt = 1.0

    Nt, Nf = 512, 128
    N = Nt * Nf

    wdm = WDM.WDM.WDM_transform(dt=dt, Nf=Nf, N=N, q=8, calc_m0=True)

    transformer = TFW.code.TD_to_TFD_transform.Transformer(wdm, 
                                                           num_freq_points=50, 
                                                           num_pixels=5)
    
    par = {'phic': 1.0, 'tc':32768.0, 'w':10000.0, 'f':0.05, 'fdot':1.0e-6}

    t_n = jnp.arange(wdm.Nt)*wdm.dT

    A_n = amplitude(t_n, par)
    Phi_n = phase(t_n, par)
    f_n = frequency(t_n, par)

    # the fast transform
    wnm = transformer.transform(A_n, Phi_n, f_n)
    wp, wc = jnp.real(wnm), jnp.imag(wnm)

    # the numerical discrete wavelet transform
    Wp = wdm(amplitude(wdm.times, par)*jnp.cos(phase(wdm.times, par)))
    Wc = wdm(amplitude(wdm.times, par)*jnp.sin(phase(wdm.times, par)))

    edge = 50
    f_low, f_high = 0.05, 0.1

    wW = wavelet_inner_product(wdm, wp, Wp, f_low=f_low, f_high=f_high, t_low=edge*wdm.dT, t_high=wdm.T-edge*wdm.dT)
    ww = wavelet_inner_product(wdm, wp, wp, f_low=f_low, f_high=f_high, t_low=edge*wdm.dT, t_high=wdm.T-edge*wdm.dT)
    WW = wavelet_inner_product(wdm, Wp, Wp, f_low=f_low, f_high=f_high, t_low=edge*wdm.dT, t_high=wdm.T-edge*wdm.dT)

    mismatch = 1.0 - wW/jnp.sqrt(ww*WW)

    assert mismatch < 1.0e-2, f"Transform failed: {mismatch=}"

