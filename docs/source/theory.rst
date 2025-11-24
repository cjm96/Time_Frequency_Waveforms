=======================
Mathematical Background
=======================

This section describes the mathematical background to the fast wavelet transform of harmonic-based GW waveform models.



.. contents::
   :local:



Introduction
------------

Gravitational waveform models are usually constructed in either the time domain 
(TD; :math:`h_+(t)`, :math:`h_\times(t)`) or the frequency domain (FD; 
:math:`\tilde{h}_+(f)`, :math:`\tilde{h}_\times(f)`).
If gravitational wave data analysis is to be performed in the time-frequency domain 
(TFD) then it is also necessary to transform our models into this TFD. 

This document describes a fast method for transforming harmonic-based waveforms from either the 
time to the time-frequency domain (TD :math:`\rightarrow` TFD) of from the 
frequency to the time-frequency domain (FD  :math:`\rightarrow` TFD).
These fast transforms rely on expanding the waveform phase (e.g.\ as a function of time) in each harmonic. 
This requires that the frequency is slowly varying over the scale of the wavelets.
These fast transforms have some similarities with the stationary phase approximation which is used 
to compute the Fourier transform between the time and frequency domains.

The next section starts by reviewing the Wilson-Daubechies-Meyer (WDM) wavelets which are used 
throughout as a basis for the TFD.  
The fast TD :math:`\rightarrow` TFD and FD :math:`\rightarrow` TFD transforms are described in 
subsequent two sections respectively.


WDM Wavelets
------------

The WDM family of wavelets are implemented in the `WDM\_GW\_wavelets` package 
`(GitHub) <https://cjm96.github.io/WDM_GW_wavelets/theory.html#/>`_;
see also Ref. [1]_ and Ref. [2]_.
The basis wavelets can be written as

.. math::

    g_{nm}(t) = 
        \begin{cases}
        \sqrt{2} \cos\left(2\pi m [t-n\Delta T] \Delta F\right) \phi(t-n\Delta T) & \mathrm{if}\;n+m\;\mathrm{even} \\
        \sqrt{2} (-1)^{nm} \sin\left(2\pi m [t-n\Delta T] \Delta F\right) \phi(t-n\Delta T) & \mathrm{if}\;n+m\;\mathrm{odd} 
    \end{cases} ,


where :math:`\phi(t)`` is a universal window function.
The :math:`n` index shifts the central time :math:`n\Delta T`` of the wavelet while the 
:math:`m` index shifts the central frequency :math:`m\Delta F`.
The wavelet time and frequency resolutions satisfy :math:`\Delta F \Delta T=1/2`.
These expressions hold for all values of :math:`n=0,1,\ldots, N_{t}-1`` but only for non-zero values of 
:math:`m=1,2,\ldots, N_{f}-1`; the :math:`m=0` wavelets store the zero and Nyquist frequency components of the signal 
and have to be handled separately.
It will also be necessary to use the dual WDM wavelet basis;

.. math::
 
    \hat{g}_{nm}(t) = 
        \begin{cases}
        \sqrt{2} \sin\left(2\pi m [t-n\Delta T] \Delta F\right) \phi(t-n\Delta T) & \mathrm{if}\;n+m\;\mathrm{even} \\
        \sqrt{2} (-1)^{nm} \cos\left(2\pi m [t-n\Delta T] \Delta F\right) \phi(t-n\Delta T) & \mathrm{if}\;n+m\;\mathrm{odd} \\
    \end{cases}.

The two bases are related by a time shift :math:`\Delta T/2`` and span exactly the same space; see the 
discussion in Ref. [1]_.


Fast Wavelet Transform (TD :math:`\rightarrow` TFD)
------------

Fast Wavelet Transform (FD :math:`\rightarrow` TFD)
------------

References
----------

.. [1] V. Necula, S. Klimenko & G. Mitselmakher, *Transient analysis with fast Wilson-Daubechies time-frequency transform*, Journal of Physics: Conference Series 363 012032, 2012.  
       `DOI 10.1088/1742-6596/363/1/012032 <https://iopscience.iop.org/article/10.1088/1742-6596/363/1/012032>`_

.. [2] N. J. Cornish, *Time-Frequency Analysis of Gravitational Wave Data*, Physical Review D 102 124038, 2020.  
       `arXiv:2009.00043 <https://arxiv.org/abs/2009.00043>`_




