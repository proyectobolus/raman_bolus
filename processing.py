"""Spectrum processing utilities."""
import numpy as np
import matplotlib.pyplot as plt
import pywt
from scipy.ndimage import uniform_filter1d


def reshape_range(x, y, range_limits, baselines=None):
    """Extract spectrum within specified range."""
    mask = (x >= range_limits[0]) & (x <= range_limits[1])
    window_x = x[mask]
    window_y = y[mask]
    
    updated_baselines = {}
    if baselines:
        for method, baseline in baselines.items():
            updated_baselines[method] = baseline[mask]
    
    return window_x, window_y, updated_baselines


def interpolate_spectrum(x, y, final_size):
    """Interpolate spectrum to specified number of points."""
    expanded_x = np.linspace(x.min(), x.max(), final_size)
    expanded_y = np.interp(expanded_x, x, y)
    return expanded_x, expanded_y


def smooth_spectrum(y, smooth_half_window=3):
    """Apply uniform filter smoothing to spectrum.
    
    Parameters
    ----------
    y : array_like
        Intensity values to smooth
    smooth_half_window : int, optional
        Half-window size for smoothing (default=3)
        Full window size will be 2*smooth_half_window+1
    
    Returns
    -------
    smoothed : ndarray
        Smoothed intensity values
    """
    if smooth_half_window <= 0:
        return y
    return uniform_filter1d(y, size=2*smooth_half_window+1)


def get_wavelet_transform(intensity, wavelet='morl', scales=None):
    """Compute continuous wavelet transform."""
    if scales is None:
        scales = np.arange(10, 100)
    coefficients, frequencies = pywt.cwt(intensity, scales, wavelet)
    return coefficients, frequencies


def plot_spectrum(x, y, title='Spectrum', color='r', size=1):
    """Plot spectrum."""
    plt.scatter(x, y, s=size, c=color)
    plt.title(title)
    plt.xlabel('Raman Shift (cm⁻¹)')
    plt.ylabel('Intensity')
    plt.show()


def plot_raw_spectrum(x, y, title='Raw Spectrum', color='b', size=1):
    """Plot raw spectrum."""
    plt.scatter(x, y, s=size, c=color)
    plt.title(title)
    plt.xlabel('Raman Shift (cm⁻¹)')
    plt.ylabel('Raw Intensity')
    plt.show()


def plot_baselines(x, y, baselines, acquired_time=''):
    """Plot all computed baselines."""
    if not baselines:
        print("No baseline corrections to plot.")
        return
    
    plt.figure(figsize=(6, 5))
    
    plt.subplot(2, 1, 1)
    plt.plot(x, y, 'k-', linewidth=1, label='Original Spectrum')
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(baselines)))
    for (method, baseline), color in zip(baselines.items(), colors):
        try:
            plt.plot(x, baseline, '--', linewidth=2, label=f'{method}', color=color)
        except ValueError:
            pass
    
    plt.xlabel('Raman Shift (cm⁻¹)')
    plt.ylabel('Intensity')
    plt.title(f'Baseline Corrections - {acquired_time}')
    plt.legend()
    
    plt.subplot(2, 1, 2)
    for (method, baseline), color in zip(baselines.items(), colors):
        corrected = y - baseline
        plt.plot(x, corrected, '-', linewidth=1, label=f'{method}', color=color)
    
    plt.xlabel('Raman Shift (cm⁻¹)')
    plt.ylabel('Baseline-Corrected Intensity')
    plt.title('Corrected Spectra')
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_wavelet_coefficients(coefficients, x_range):
    """Plot wavelet coefficients."""
    plt.imshow(coefficients, extent=[x_range[0], x_range[1], 1, coefficients.shape[0]],
               cmap='viridis', aspect='auto',
               vmax=abs(coefficients).max(),
               vmin=-abs(coefficients).max(),
               origin='lower')
    plt.ylabel('Scale')
    plt.xlabel('Raman Shift (cm⁻¹)')
    plt.title('Continuous Wavelet Transform Coefficients')
    plt.colorbar(label='Coefficient Magnitude')
    plt.show()


def return_near_value(normdictcoords, target_x):
    """Return nearest y value to target x."""
    keys = np.array(list(normdictcoords.keys()))
    idx = (np.abs(keys - target_x)).argmin()
    nearest_x = keys[idx]
    return normdictcoords[nearest_x]
