"""
Utility functions for TimeFrequencyWaveforms.

This module contains general utility functions for time-frequency analysis
of gravitational waveforms.
"""

import jax.numpy as jnp

import WDM


def example_function():
    """
    Example utility function.

    Returns
    -------
    str
        A placeholder message.
    """
    return "Utility function placeholder"


def coeffs(wdm, f, fdot=0.0, fddot=0.0):
    """
    """
    gnm = wdm.gnm_basis_allm()
    gnm_comp = wdm.gnm_basis_comp_allm()

    n_vals = jnp.arange(0, wdm.Nt)

    t_minus_tn = wdm.times[:,jnp.newaxis]-n_vals[jnp.newaxis,:] * wdm.dT

    X = 2. * jnp.pi * t_minus_tn * f + \
            jnp.pi * t_minus_tn**2 * fdot + \
                (1./.3) * jnp.pi * t_minus_tn**3 * fddot
    
    cnm = wdm.dt * jnp.sum(jnp.cos(X)[:,:,jnp.newaxis] * gnm, axis=0)
    snm = wdm.dt * jnp.sum(jnp.sin(X)[:,:,jnp.newaxis] * gnm, axis=0)
    cnm_comp = wdm.dt * jnp.sum(jnp.cos(X)[:,:,jnp.newaxis] * gnm_comp, axis=0)
    snm_comp = wdm.dt * jnp.sum(jnp.sin(X)[:,:,jnp.newaxis] * gnm_comp, axis=0)

    return cnm, snm, cnm_comp, snm_comp