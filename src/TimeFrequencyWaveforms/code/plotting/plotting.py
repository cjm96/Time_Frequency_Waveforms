"""
Plotting functions for TimeFrequencyWaveforms.

This module contains functions for visualizing time-frequency representations
of gravitational waveforms.
"""

import matplotlib.pyplot as plt
import jax.numpy as jnp


def example_plot():
    """
    Example plotting function.

    Returns
    -------
    tuple
        A matplotlib figure and axis object.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlabel('Time')
    ax.set_ylabel('Amplitude')
    ax.set_title('Example Plot')
    return fig, ax
