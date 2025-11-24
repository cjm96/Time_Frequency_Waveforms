import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from functools import partial

import WDM


@partial(jax.jit, static_argnums=0)
def Xn(wdm : WDM.WDM.WDM_transform, 
       n : int, 
       m : int, 
       f : float, 
       fdot : float=0.0, 
       fddot : float=0.0) -> jnp.ndarray:
    r"""
    Compute the phase term :math:`X_n(t)` at the sample times stored in the wdm 
    object.

    .. math::

        X_n(f, \dot{f}, \ldots) = 2\pi (t-n\Delta T) f + \
                                    \pi (t-n\Delta T)^2 \dot{f} + \
                                        (1/3) \pi (t-n\Delta T)^3 \ddot{f} .

    Parameters
    ----------
    wdm : WDM.WDM.WDM_transform
        An instance of the WDM wavelet transform class.
    n : int
        Wavelet time index.
    m : int
        Wavelet frequency index.
    f : float
        Frequency at which to evaluate the phase term [Hz].
    fdot : float
        Frequency derivative [Hz/s]. Optional.
    fddot : float
        Second frequency derivative [Hz/s/s]. Optional.

    Returns
    -------
    Xn : jnp.ndarray 
        The phase term, shape=(N,).
    """
    Xn = 2. * jnp.pi * (wdm.times-n*wdm.dT) * f + \
            jnp.pi * (wdm.times-n*wdm.dT)**2 * fdot + \
                (1./.3) * jnp.pi * (wdm.times-n*wdm.dT)**3 * fddot  
    return Xn


def cnm(wdm : WDM.WDM.WDM_transform,
        n : int, 
        m : int,
        f : float,
        fdot : float=0.0,
        fddot : float=0.0) -> jnp.ndarray:
    r"""
    Compute the coefficient :math:`c_{nm}(f,\dot{f},\ldots)`.

    .. math::

        c_{nm}(f,\dot{f},\ldots) = \int\mathrm{d}t\; \
                                        \cos X_n(f,\dot{f},\ldots) g_{nm}(t).

    Parameters
    ----------
    wdm : WDM.WDM.WDM_transform
        An instance of the WDM wavelet transform class.
    n : int
        Wavelet time index.
    m : int
        Wavelet frequency index.
    f : float
        Frequency at which to evaluate the phase term [Hz].
    fdot : float
        Frequency derivative [Hz/s]. Optional.
    fddot : float
        Second frequency derivative [Hz/s/s]. Optional.

    Returns
    -------
    c : jnp.ndarray 
        The phase term, shape=(N,).
    """
    g = wdm.gnm(n, m)
    X = Xn(wdm, n, m, f, fdot=fdot, fddot=fddot)  
    c = wdm.dt * jnp.sum(jnp.cos(X)*g)
    return c


def snm(wdm : WDM.WDM.WDM_transform,
        n : int, 
        m : int,
        f : float,
        fdot : float=0.0,
        fddot : float=0.0) -> jnp.ndarray:
    r"""
    Compute the coefficient :math:`s_{nm}(f,\dot{f},\ldots)`.

    .. math::

        s_{nm}(f,\dot{f},\ldots) = \int\mathrm{d}t\; \
                                        \sin X_n(f,\dot{f},\ldots) g_{nm}(t).

    Parameters
    ----------
    wdm : WDM.WDM.WDM_transform
        An instance of the WDM wavelet transform class.
    n : int
        Wavelet time index.
    m : int
        Wavelet frequency index.
    f : float
        Frequency at which to evaluate the phase term [Hz].
    fdot : float
        Frequency derivative [Hz/s]. Optional.
    fddot : float
        Second frequency derivative [Hz/s/s]. Optional.

    Returns
    -------
    s : jnp.ndarray 
        The phase term, shape=(N,).
    """
    g = wdm.gnm(n, m)
    X = Xn(wdm, n, m, f, fdot=fdot, fddot=fddot)     
    s = wdm.dt * jnp.sum(jnp.sin(X)*g)
    return s


def chatnm(wdm : WDM.WDM.WDM_transform,
           n : int,
           m : int,
           f : float,
           fdot : float=0.0,
           fddot : float=0.0) -> jnp.ndarray:
    r"""
    Compute the coefficient :math:`\hat{c}_{nm}(f,\dot{f},\ldots)`.

    .. math::

        \hat{c}_{nm}(f,\dot{f},\ldots) = \int\mathrm{d}t\; \
                                    \cos X_n(f,\dot{f},\ldots) \hat{g}_{nm}(t).

    Parameters
    ----------
    wdm : WDM.WDM.WDM_transform
        An instance of the WDM wavelet transform class.
    n : int
        Wavelet time index.
    m : int
        Wavelet frequency index.
    f : float
        Frequency at which to evaluate the phase term [Hz].
    fdot : float
        Frequency derivative [Hz/s]. Optional.
    fddot : float
        Second frequency derivative [Hz/s/s]. Optional.

    Returns
    -------
    chat : jnp.ndarray 
        The phase term, shape=(N,).
    """
    ghat = wdm.gnm_dual(n, m)
    X = Xn(wdm, n, m, f, fdot=fdot, fddot=fddot)   
    chat = wdm.dt * jnp.sum(jnp.cos(X)*ghat)
    return chat


def shatnm(wdm : WDM.WDM.WDM_transform,
           n : int,
           m : int,
           f : float,
           fdot : float=0.0,
           fddot : float=0.0) -> jnp.ndarray:
    r"""
    Compute the coefficient :math:`\hat{s}_{nm}(f,\dot{f},\ldots)`.

    .. math::

        \hat{s}_{nm}(f,\dot{f},\ldots) = \int\mathrm{d}t\; \
                                    \sin X_n(f,\dot{f},\ldots) \hat{g}_{nm}(t).

    Parameters
    ----------
    wdm : WDM.WDM.WDM_transform
        An instance of the WDM wavelet transform class.
    n : int
        Wavelet time index.
    m : int
        Wavelet frequency index.
    f : float
        Frequency at which to evaluate the phase term [Hz].
    fdot : float
        Frequency derivative [Hz/s]. Optional.
    fddot : float
        Second frequency derivative [Hz/s/s]. Optional.

    Returns
    -------
    shat : jnp.ndarray 
        The phase term, shape=(N,).
    """
    ghat = wdm.gnm_dual(n, m)
    X = Xn(wdm, n, m, f, fdot=fdot, fddot=fddot)   
    shat = wdm.dt * jnp.sum(jnp.sin(X)*ghat)
    return shat


@jax.jit
def row_roll(A: jnp.ndarray, 
             shifts: jnp.ndarray) -> jnp.ndarray:
    """
    Roll each row of a 2D array by a different integer amount along axis 1.

    Given input array A of shape (N, M) and a vector of integer shifts of shape 
    (N,), this function circularly shifts (or rolls) the elements of row i of A 
    by shift[i] positions along the second axis. 

    Parameters
    ----------
    A : jnp.ndarray
        Input array, shape=(N, M).
    shifts : jnp.ndarray
        Integer array of shifts, shape=(N,), dtype=int.

    Returns
    -------
    B : jnp.ndarray
        Output array, shape=(N, M).
    """
    N, M = A.shape
    cols = jnp.arange(M)
    idx = (cols[None, :] - shifts[:, None]) % M
    B = A[jnp.arange(N)[:, None], idx]
    return B