import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax.scipy.interpolate import RegularGridInterpolator

from typing import Tuple
from functools import partial

from TimeFrequencyWaveforms.code import utils

import WDM


class Transformer:
    """
    Description. 
    """

    def __init__(self, 
                 wdm : WDM.WDM.WDM_transform,
                 num_freq_points : int=100,
                 fdot_grid_spec : tuple=None,
                 fddot_grid_spec : tuple=None,
                 num_pixels : int=None) -> None:
        """
        Parameters
        ----------
        wdm : WDM.WDM
            An instance of the WDM wavelet transform class.
        num_freq_points : int
            Number of frequency points in the interpolation grid. Optional.
        fdot_grid_spec : tuple
            Specification for fdot grid as (min, max, num_points). Optional.
            Default is None, meaning no interpolation over fdot.
        fddot_grid_spec : tuple
            Specification for fddot grid as (min, max, num_points). Optional.
            Default is None, meaning no interpolation over fddot.
        num_pixels : int
            Number of time pixels to interpolate. If None, then use all Nf 
            pixels. Optional.

        Returns
        -------
        None
        """
        self.wdm = wdm

        self.num_freq_points = num_freq_points

        self.n_ref = 2 * ( self.wdm.Nt // 4 )
        self.m_ref = self.wdm.Nf // 2

        self.f_ref = self.m_ref * self.wdm.dF
        self.f_grid_spec = (0., 2*self.wdm.dF, self.num_freq_points)
        self.fdot_grid_spec = fdot_grid_spec
        self.fddot_grid_spec = fddot_grid_spec

        if self.fddot_grid_spec is not None:
            assert self.fdot_grid_spec is not None, \
                "fdot_grid_spec must be provided if fddot_grid_spec is."

        self.n_vals = jnp.arange(0, self.wdm.Nt)
        self.m_vals = jnp.arange(0, self.wdm.Nf)

        self.alt = (-1)**(self.n_vals[:,jnp.newaxis]+self.m_vals[jnp.newaxis,:])

        self.num_pixels = num_pixels if num_pixels is not None else self.wdm.Nf
        assert self.num_pixels <= self.wdm.Nf, \
            "num_pixels cannot be larger than wdm.Nf."
        self.m_pixel_range = self.m_ref - \
                                self.num_pixels//2 + \
                                    jnp.arange(self.num_pixels)
        if self.num_pixels == self.wdm.Nf:
            assert jnp.all(jnp.equal(self.m_pixel_range, self.m_vals)), \
                "I've messed up!"

        self.mask_n_even = (self.n_vals % 2) == 0
        self.mask_n_even = jnp.outer(self.mask_n_even, 
                                     jnp.ones(len(self.m_vals), dtype=bool))

        self.grids = self.make_grids()
        self.grid_shape = tuple(grid.shape[0] for grid in self.grids)
        self.dim = len(self.grids)

        (self.cnm_interp, 
         self.snm_interp, 
         self.chatnm_interp, 
         self.shatnm_interp) = self.make_interpolators()

    def make_grids(self) -> jnp.ndarray:
        """
        Make the regular grid for interpolation.

        Returns
        -------
        grids : jnp.ndarray
            The grid points for interpolation, shape=(num_freq_points, ...).
        """
        dim = 1
        f_grid = jnp.linspace(*self.f_grid_spec)
        grids = (f_grid, )
        
        if self.fdot_grid_spec is not None:
            dim += 1
            fdot_grid = jnp.linspace(*self.fdot_grid_spec)
            grids += (fdot_grid, )

        if self.fddot_grid_spec is not None:
            dim += 1
            fddot_grid = jnp.linspace(*self.fddot_grid_spec)
            grids += (fddot_grid, )

        return grids
    
    def make_interpolators(self) -> Tuple[RegularGridInterpolator,
                                          RegularGridInterpolator,
                                          RegularGridInterpolator,
                                          RegularGridInterpolator]:
        """
        Create interpolators for the coefficients cnm, snm, chatnm and shatnm.

        Returns
        -------
        interpolators : tuple
            Four interpolators for cnm, snm, chatnm, shatnm. 
        """
        cos_data     = jnp.zeros(self.grid_shape+(self.num_pixels,))
        sin_data     = jnp.zeros(self.grid_shape+(self.num_pixels,))
        cos_hat_data = jnp.zeros(self.grid_shape+(self.num_pixels,))
        sin_hat_data = jnp.zeros(self.grid_shape+(self.num_pixels,))

        for i, f in enumerate(self.grids[0]+self.f_ref):
            if self.dim == 1:
                fdot = 0.0
                fddot = 0.0
                for m_, m in enumerate(self.m_pixel_range):
                    cos_data     = cos_data.at[i,m_].set(utils.cnm(self.wdm, 
                                            self.n_ref, m, f, fdot, fddot))
                    sin_data     = sin_data.at[i,m_].set(utils.snm(self.wdm, 
                                            self.n_ref, m, f, fdot, fddot))
                    cos_hat_data = cos_hat_data.at[i,m_].set(utils.chatnm(self.wdm, 
                                            self.n_ref, m, f, fdot, fddot))
                    sin_hat_data = sin_hat_data.at[i,m_].set(utils.shatnm(self.wdm, 
                                            self.n_ref, m, f, fdot, fddot))
            elif self.dim == 2:
                for j, fdot in enumerate(self.grids[1]):
                    fddot = 0.0
                    for m_, m in enumerate(self.m_pixel_range):
                        cos_data     = cos_data.at[i,j,m_].set(utils.cnm(self.wdm, 
                                                self.n_ref, m, f, fdot, fddot))
                        sin_data     = sin_data.at[i,j,m_].set(utils.snm(self.wdm, 
                                                self.n_ref, m, f, fdot, fddot))
                        cos_hat_data = cos_hat_data.at[i,j,m_].set(utils.chatnm(
                                                self.wdm, self.n_ref, m, 
                                                f, fdot, fddot))
                        sin_hat_data = sin_hat_data.at[i,j,m_].set(utils.shatnm(
                                                self.wdm, self.n_ref, m, 
                                                f, fdot, fddot))
            elif self.dim == 3:
                for j, fdot in enumerate(self.grids[1]):
                    for k, fddot in enumerate(self.grids[2]):
                        for m_, m in enumerate(self.m_pixel_range):
                            cos_data     = cos_data.at[i,j,k,m_].set(utils.cnm(
                                                    self.wdm, self.n_ref, m, 
                                                    f, fdot, fddot))
                            sin_data     = sin_data.at[i,j,k,m_].set(utils.snm(
                                                    self.wdm, self.n_ref, m, 
                                                    f, fdot, fddot))
                            cos_hat_data = cos_hat_data.at[i,j,k,m_].set(utils.chatnm(
                                                    self.wdm, self.n_ref, m, 
                                                    f, fdot, fddot))
                            sin_hat_data = sin_hat_data.at[i,j,k,m_].set(utils.shatnm(
                                                    self.wdm, self.n_ref, m, 
                                                    f, fdot, fddot))

        cnm_interp = RegularGridInterpolator(self.grids, cos_data, 
                                             method='linear', 
                                             bounds_error=False, 
                                             fill_value=0.0)
        snm_interp = RegularGridInterpolator(self.grids, sin_data, 
                                             method='linear', 
                                             bounds_error=False, 
                                             fill_value=0.0)
        chatnm_interp = RegularGridInterpolator(self.grids, cos_hat_data, 
                                             method='linear', 
                                             bounds_error=False, 
                                             fill_value=0.0)
        shatnm_interp = RegularGridInterpolator(self.grids, sin_hat_data, 
                                             method='linear', 
                                             bounds_error=False, 
                                             fill_value=0.0)
        
        interpolators = (cnm_interp, snm_interp, chatnm_interp, shatnm_interp)

        return interpolators
    
    @partial(jax.jit, static_argnums=0)
    def coeffs(self, 
               F : jnp.ndarray, 
               fdot : jnp.ndarray=None, 
               fddot : jnp.ndarray=None) -> Tuple[jnp.ndarray,
                                                  jnp.ndarray,
                                                  jnp.ndarray,
                                                  jnp.ndarray]:
        """
        Interpolate to get the coefficients at given parameters.

        Parameters
        ----------
        F : jnp.ndarray
            Frequencies, shape=(Nt,).
        fdot : jnp.ndarray
            Frequency derivatives, shape=(Nt,).
        fddot : jnp.ndarray
            Frequency second derivatives, shape=(Nt,).

        Returns
        -------
        values : tuple
            The quantities. Each of these are jnp.ndarray with shape=(Nt, Nf).
        """
        query = jnp.array([x for x in [F, fdot, fddot] if x is not None]).T

        CNM = jnp.zeros((self.wdm.Nt, self.wdm.Nf))
        SNM = jnp.zeros((self.wdm.Nt, self.wdm.Nf))
        CHATNM = jnp.zeros((self.wdm.Nt, self.wdm.Nf))
        SHATNM = jnp.zeros((self.wdm.Nt, self.wdm.Nf))

        CNM = CNM.at[:,self.m_pixel_range].set(self.cnm_interp(query))
        SNM = SNM.at[:,self.m_pixel_range].set(self.snm_interp(query))
        CHATNM = CHATNM.at[:,self.m_pixel_range].set(self.chatnm_interp(query))
        SHATNM = SHATNM.at[:,self.m_pixel_range].set(self.shatnm_interp(query))

        #CNM = self.cnm_interp(query)
        #SNM = self.snm_interp(query)
        #CHATNM = self.chatnm_interp(query)
        #SHATNM = self.shatnm_interp(query)

        CNM_shifted = jnp.where(self.mask_n_even, CNM, CHATNM)
        SNM_shifted = jnp.where(self.mask_n_even, SNM, SHATNM)
        CHATNM_shifted = jnp.where(self.mask_n_even, CHATNM, CNM)
        SHATNM_shifted = jnp.where(self.mask_n_even, SHATNM, SNM)

        values = (CNM_shifted, SNM_shifted, CHATNM_shifted, SHATNM_shifted)

        return values
    
    @partial(jax.jit, static_argnums=0)
    def cnm_snm(self, 
                A_n : jnp.ndarray, 
                phi_n : jnp.ndarray, 
                f_n : jnp.ndarray, 
                fdot_n : jnp.ndarray=None, 
                fddot_n : jnp.ndarray=None) -> Tuple[jnp.ndarray,
                                                    jnp.ndarray,
                                                    jnp.ndarray,
                                                    jnp.ndarray]:
        r"""
        Transformer.

        Parameters
        ----------
        A_n : jnp.ndarray
            The waveform amplitude :math:`A(t_n)` evaluated at the sparse 
            wavelet times :math:`t_n = n \Delta T`.
        phi_n : jnp.ndarray
            The waveform phase :math:`\Phi(t_n)` [rad] evaluated at the sparse 
            wavelet times :math:`t_n = n \Delta T`.
        f_n : jnp.ndarray
            The waveform frequency :math:`f(t_n)` [Hz] evaluated at the sparse 
            wavelet times :math:`t_n = n \Delta T`.
        fdot_n : jnp.ndarray
            The frequency derivative :math:`\dot{f}(t_n)` [Hz/s] 
            evaluated at the sparse wavelet times :math:`t_n = n \Delta T`.
        fddot : jnp.ndarray
            The frequency second derivative :math:`\ddot{f}(t_n)` [Hz/s] 
            evaluated at the sparse wavelet times :math:`t_n = n \Delta T`.

        Returns
        -------
        CNM, SNM, CHATNM, SHATNM
        """
        F_n_minus_f_ref = (f_n-self.f_ref)%(2*self.wdm.dF)
        z_n = jnp.floor((f_n-self.f_ref)/(2*self.wdm.dF)).astype(int)

        CNM, SNM, CHATNM, SHATNM = self.coeffs(F_n_minus_f_ref, 
                                               fdot_n, fddot_n)
        
        return CNM, SNM, CHATNM, SHATNM
    
    @partial(jax.jit, static_argnums=0)
    def transform(self, 
                  A_n : jnp.ndarray, 
                  phi_n : jnp.ndarray, 
                  f_n : jnp.ndarray, 
                  fdot_n : jnp.ndarray=None, 
                  fddot_n : jnp.ndarray=None) -> jnp.ndarray:
        r"""
        Transformer.

        Parameters
        ----------
        A_n : jnp.ndarray
            The waveform amplitude :math:`A(t_n)` evaluated at the sparse 
            wavelet times :math:`t_n = n \Delta T`.
        phi_n : jnp.ndarray
            The waveform phase :math:`\Phi(t_n)` [rad] evaluated at the sparse 
            wavelet times :math:`t_n = n \Delta T`.
        f_n : jnp.ndarray
            The waveform frequency :math:`f(t_n)` [Hz] evaluated at the sparse 
            wavelet times :math:`t_n = n \Delta T`.
        fdot_n : jnp.ndarray
            The frequency derivative :math:`\dot{f}(t_n)` [Hz/s] 
            evaluated at the sparse wavelet times :math:`t_n = n \Delta T`.
        fddot : jnp.ndarray
            The frequency second derivative :math:`\ddot{f}(t_n)` [Hz/s] 
            evaluated at the sparse wavelet times :math:`t_n = n \Delta T`.

        Returns
        -------
        wnm : jnp.ndarray
            The wavelet coefficients of the waveform; the real part if the plus
            polarisation, the imaginary part is the cross polarisation.
            Array shape=(self.wdm.Nt, self.wdm.Nf), dtype=complex.
        """
        F_n_minus_f_ref = (f_n-self.f_ref)%(2*self.wdm.dF)
        z_n = jnp.floor((f_n-self.f_ref)/(2*self.wdm.dF)).astype(int)

        CNM, SNM, CHATNM, SHATNM = self.coeffs(F_n_minus_f_ref, 
                                               fdot_n, fddot_n)
        
        cnm = 0.5 * ( utils.row_roll(CNM, +2*z_n) +
                      utils.row_roll(CNM, -2*z_n) +
                      self.alt * utils.row_roll(SHATNM, +2*z_n) -
                      self.alt * utils.row_roll(SHATNM, -2*z_n) ) 
        
        snm = 0.5 * ( utils.row_roll(SNM, +2*z_n) +
                      utils.row_roll(SNM, -2*z_n) -
                      self.alt * utils.row_roll(CHATNM, +2*z_n) +
                      self.alt * utils.row_roll(CHATNM, -2*z_n) ) 

        wnm = ( A_n[:,jnp.newaxis] * \
                jnp.exp(1j*phi_n[:,jnp.newaxis]) * \
                    ( cnm + (1j) * snm ) )

        return wnm
