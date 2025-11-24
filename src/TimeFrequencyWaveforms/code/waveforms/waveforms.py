"""
Waveform processing functions for TimeFrequencyWaveforms.

This module contains functions for processing gravitational waveforms
in time and frequency domains using WDM wavelet transforms.
"""

import jax.numpy as jnp

from jax.scipy.interpolate import RegularGridInterpolator

from ..utils.utils import coeffs



def linear_interp_coeff_data_ndim(grids, data, query, fill_value=0.0):
    """
    D-dimensional interpolation of the cnm, snm coefficient matrix-valued data. 
    Simple, jax-compatible, D-dimensional linear interpolation that works with 
    data that is regularly sampled along each axis. The interpolation is 
    queried at a different point for each row (n-axis) of the data.

    Parameters
    ----------
    grids : tuple, length dim
        E.g. (grid_0, grid_1, ...) where grids_d is a 1D increasing and 
        regularly spaced jax array of shape (num_pts_d,) for dimension d.
    data : jnp.ndarray
        Values to be interpolated, shape (num_pts_0, num_pts_1, ..., N, M).
    query : jnp.ndarray
        Query locations, shape (dim, N).
    fill_value : float, optional
        Value used when query is outside grid range.

    Returns
    -------
    y : jnp.ndarray
        Interpolated values at query points, shape (N,M).
    """
    dim = len(grids)
    grid_shape = tuple(g.shape[0] for g in grids)

    data = jnp.asarray(data)
    N, M = data.shape[-2], data.shape[-1]
    assert data[...,0,0].shape == grid_shape, \
            "data must match shape of grids"

    grid_spacing = tuple(g[1] - g[0] for g in grids)
    for g, grid in enumerate(grids):
        assert jnp.allclose(jnp.diff(grid), grid_spacing[g]), \
            "grid is not regularly spaced along dimension {}.".format(g)

    query = jnp.asarray(query)
    assert query.shape == (dim, N), \
            "query dimension {} does not match ({}, {}).".format(query.shape, dim, N)

    # ----- per-axis: left cell indices and fractional weights t in [0,1] -----
    # Using arithmetic (regular grid): u = (x - g0)/dx, i = floor(u), t = frac(u)
    idx_list = []
    t_list = []
    inb_list = []

    for d in range(dim):
        g = grids[d]                                 # (Sd,)
        Sd = g.shape[0]
        g0 = g[0]
        dx = g[1] - g[0]                             # regular spacing
        # (D, N) -> (N,) for this axis
        qd = query[d, :]                             # (N,)

        # continuous coordinate in grid units
        u = (qd - g0) / grid_spacing[d]             # (N,)
        i = jnp.floor(u).astype(jnp.int32)          # left index
        # clamp to interior so i+1 is valid
        i = jnp.clip(i, 0, grid_shape[d] - 2)
        t = u - i                                    # fractional part in [0,1) ideally

        idx_list.append(i)                           # (N,)
        t_list.append(t)                             # (N,)
        inb_list.append((qd >= g0) & (qd <= g[-1]))  # (N,)

    idx = jnp.stack(idx_list, axis=0)                # (D, N)
    t   = jnp.stack(t_list,   axis=0)                # (D, N)
    in_bounds = jnp.logical_and.reduce(jnp.stack(inb_list, axis=0), axis=0)  # (N,)

    # ----- enumerate 2^D corners (no Python loop over corners) -----
    corners = jnp.stack(jnp.indices((2,) * dim), axis=-1).reshape(-1, dim)  # (C, dim)
    C = corners.shape[0]

    # Corner indices for each n: (C, dim, N)
    corner_idx = idx[None, :, :] + corners[:, :, None]

    # Corner weights for each n: (C, N)
    factors = jnp.where(corners[:, :, None] == 1, t[None, :, :], 1.0 - t[None, :, :])
    w = jnp.prod(factors, axis=1)  # (C, N)

    # ----- gather all corner matrices at once -----
    P = int(jnp.prod(jnp.array(grid_shape)))
    values_flat = data.reshape(P, N, M)                                   # (P, N, M)

    # ravel (C, D, N) -> (C, N) linear indices into first D dims
    p = jnp.ravel_multi_index(corner_idx.transpose(1, 0, 2),
                              dims=grid_shape)                       # (C, N)

    # Gather corner matrices for each (c, n): (C, N, M)
    corner_vals = values_flat[p, jnp.arange(N)[None, :], :]               # (C, N, M)

    # Weighted sum over corners -> (N, M)
    y = jnp.sum(w[:, :, None] * corner_vals, axis=0)                    # (N, M)

    # ----- out-of-bounds fill (row-wise) -----
    y = jnp.where(in_bounds[:, None], y, fill_value)

    return y



def example_waveform_function():
    """
    Example waveform processing function.

    Returns
    -------
    str
        A placeholder message.
    """
    return "Waveform processing function placeholder"


class Transformer:

    def __init__(self, wdm, 
                 num_freq_points=100,
                 c=None, 
                 fdot_grid_spec=None,
                 fddot_grid_spec=None,
                 calc_m0=False):
        """
        Parameters
        ----------
        wdm : WDM.WDM
            An instance of the WDM wavelet transform class.
        fdot_grid_spec : tuple, optional
            Specification for the fdot grid as (min, max, num_points).
        fddot_grid_spec : tuple, optional
            Specification for the fddot grid as (min, max, num_points).
        """
        self.wdm = wdm

        self.num_freq_points = num_freq_points
        self.fdot_grid_spec = fdot_grid_spec
        self.fddot_grid_spec = fddot_grid_spec

        self.calc_m0 = calc_m0

        self.grid, self.dim = self.make_grid()

        self.cnm_data, self.snm_data, self.cnm_comp_data, self.snm_comp_data = self.make_data()


    def make_grid(self):
        """
        Make the regular grid for interpolation.
        """
        dim = 1
        f_grid = jnp.linspace(0, 2*self.wdm.dF, self.num_freq_points)
        grid = (f_grid, )
        
        if self.fdot_grid_spec is not None:
            dim += 1
            fdot_grid = jnp.linspace(*self.fdot_grid_spec)
            grid += (fdot_grid, )

        if self.fddot_grid_spec is not None:
            dim += 1
            fddot_grid = jnp.linspace(*self.fddot_grid_spec)
            grid += (fddot_grid, )

        return grid, dim
    

    def make_data(self):
        """
        Make the data to be interpolated.
        """
        data_shape = (self.num_freq_points, )
        if self.fdot_grid_spec is not None:
            data_shape += (self.fdot_grid_spec[2], )
        if self.fddot_grid_spec is not None:
            data_shape += (self.fddot_grid_spec[2], )

        data_shape += (self.wdm.Nt, 2*self.wdm.Nf)

        cnm_data = jnp.zeros(data_shape)
        snm_data = jnp.zeros(data_shape)
        cnm_comp_data = jnp.zeros(data_shape)
        snm_comp_data = jnp.zeros(data_shape)

        if self.fdot_grid_spec is None:
            
            for i, f in enumerate(self.grid[0]):
                cnm, snm, cnm_comp, snm_comp = coeffs(self.wdm, f)

                cnm_data = cnm_data.at[i].set(cnm)
                snm_data = snm_data.at[i].set(snm)
                cnm_comp_data = cnm_comp_data.at[i].set(cnm_comp)
                snm_comp_data = snm_comp_data.at[i].set(snm_comp)

        else:
            
            if self.fddot_grid_spec is None:

                for i, f in enumerate(self.grid[0]):
                    for j, fdot in enumerate(self.grid[1]):
                        cnm, snm, cnm_comp, snm_comp = coeffs(self.wdm, f, fdot=fdot)

                        cnm_data = cnm_data.at[i,j].set(cnm)
                        snm_data = snm_data.at[i,j].set(snm)
                        cnm_comp_data = cnm_comp_data.at[i,j].set(cnm_comp)
                        snm_comp_data = snm_comp_data.at[i,j].set(snm_comp)

            else:

                for i, f in enumerate(self.grid[0]):
                    for j, fdot in enumerate(self.grid[1]):
                        for k, fddot in enumerate(self.grid[2]):
                            cnm, snm, cnm_comp, snm_comp = coeffs(self.wdm, f, fdot=fdot, fddot=fddot)
                            
                            cnm_data = cnm_data.at[i,j,k].set(cnm)
                            snm_data = snm_data.at[i,j,k].set(snm)
                            cnm_comp_data = cnm_comp_data.at[i,j,k].set(cnm_comp)
                            snm_comp_data = snm_comp_data.at[i,j,k].set(snm_comp)

        return cnm_data, snm_data, cnm_comp_data, snm_comp_data

    def coeffs(self, f, fdot=None, fddot=None):
        """
        Interpolate to get the waveform coefficients at given parameters.
        """
        n_vals = jnp.arange(0, self.wdm.Nt, dtype=jnp.int32)
        m_vals = jnp.arange(0, 2*self.wdm.Nf, dtype=jnp.int32)

        Df2 = 2 * self.wdm.dF 

        z = jnp.array(f // Df2, dtype=jnp.int32)
        f_shifted = f % Df2

        query = jnp.array([x for x in [f_shifted, fdot, fddot] if x is not None])

        cnm = linear_interp_coeff_data_ndim(self.grid, self.cnm_data, query)
        snm = linear_interp_coeff_data_ndim(self.grid, self.snm_data, query)
        cnm_comp = linear_interp_coeff_data_ndim(self.grid, self.cnm_comp_data, query)
        snm_comp = linear_interp_coeff_data_ndim(self.grid, self.snm_comp_data, query)

        idx_minus_2z = (m_vals[None, :] - 2*z[:, None]) % (2*self.wdm.Nf)
        idx_pluus_2z = (m_vals[None, :] + 2*z[:, None]) % (2*self.wdm.Nf)  
    
        cnm_minus_2z = cnm[n_vals[:, None], idx_minus_2z]
        cnm_plus_2z = cnm[n_vals[:, None], idx_pluus_2z]
        snm_minus_2z = snm[n_vals[:, None], idx_minus_2z]
        snm_plus_2z = snm[n_vals[:, None], idx_pluus_2z]
        cnm_comp_minus_2z = cnm_comp[n_vals[:, None], idx_minus_2z]
        cnm_comp_plus_2z = cnm_comp[n_vals[:, None], idx_pluus_2z]
        snm_comp_minus_2z = snm_comp[n_vals[:, None], idx_minus_2z]
        snm_comp_plus_2z = snm_comp[n_vals[:, None], idx_pluus_2z]

        cnm = 0.5 * ( cnm_minus_2z + 
                      cnm_plus_2z + 
                      (-1.)**(n_vals[:,jnp.newaxis]+m_vals[jnp.newaxis,:]) * ( 
                            snm_comp_minus_2z - 
                            snm_comp_plus_2z
                            )
                    )
        
        snm = 0.5 * ( snm_minus_2z + 
                      snm_plus_2z -
                      (-1.)**(n_vals[:,jnp.newaxis]+m_vals[jnp.newaxis,:]) * ( 
                            cnm_comp_minus_2z - 
                            cnm_comp_plus_2z
                            )
                    )

        return cnm[:,:self.wdm.Nf], snm[:,:self.wdm.Nf]
    

    def transform(self, amp, phase, freq, fdot=None):
        """
        """

        cnm, snm = self.coeffs(freq, fdot=fdot)
        
        wnm = amp[:,jnp.newaxis] * jnp.exp((1j)*phase[:,jnp.newaxis]) * ( cnm + (1j)*snm )

        if self.calc_m0:
            pass

        return wnm

    
