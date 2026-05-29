"""RamanSpectrum v2 - Refactored spectrum analysis class."""
from .data_loader import load_spectrum_file, process_metadata, normalize_spectrum, create_coordinate_dicts
from .baseline_correction import apply_baseline_correction, get_baseline_summary
from .processing import (
    reshape_range, interpolate_spectrum, get_wavelet_transform,
    plot_spectrum, plot_raw_spectrum, plot_baselines, plot_wavelet_coefficients,
    return_near_value, smooth_spectrum
)


class RamanSpectrum:
    """Class for analyzing Raman spectra data (v2 - Modular)."""
    
    def __repr__(self):
        return f'RamanSpectrum: {self.filepath}'
    
    def __str__(self):
        return f'Raman Spectrum from {self.acquired}'
    
    def __init__(self, filepath):
        self.batch_id = None
        self.filepath = filepath
        self.metadata = {}
        self.recoincidences = ""
        self.retagone = ""
        self.retagtwo = ""
        
        # Initialize baseline storage
        self.baselines = {}
        self.baseline_errors = {}
        self.baseline_methods_used = []
        self.current_baseline = None
        self.baseline_wavenumbers = None
        self.baseline_corrected_intensity = None
        self.current_baseline_method = None
        
        # Load data
        self.metadata, self.x, self.y, self.data = load_spectrum_file(filepath)
        self.raw_y = self.y.copy()
        
        # Process metadata
        processed = process_metadata(self.metadata)
        self.acquired = processed['acquired']
        self.date_acquired = processed['date_acquired']
        self.time_acquired = processed['time_acquired']
        self.title = processed['title']
        self.xmin = processed['xmin']
        self.xmax = processed['xmax']
        self.total_range = processed['total_range']
        self.num_points = len(self.x)
        
        # Normalize and create coordinate dicts
        self.y = normalize_spectrum(self.y)
        self.dictcoords, self.normdictcoords = create_coordinate_dicts(self.data, self.y)
    
    def baseline_correction(self, method='poly3', smooth=True, smooth_half_window=3, **kwargs):
        """Apply baseline correction with optional smoothing.
        
        Parameters
        ----------
        method : str, optional
            Baseline correction method (default='poly3')
        smooth : bool, optional
            Whether to apply smoothing before baseline correction (default=True)
        smooth_half_window : int, optional
            Half-window size for smoothing (default=3)
        **kwargs : dict
            Additional arguments passed to baseline correction method
        """
        if hasattr(self, 'window_wavenumbers') and hasattr(self, 'window_intensity'):
            x = self.window_wavenumbers
            y = self.window_intensity.copy()
        else:
            x = self.x
            y = self.y.copy()
        
        # Apply smoothing if requested
        if smooth:
            y = smooth_spectrum(y, smooth_half_window=smooth_half_window)
        
        result = apply_baseline_correction(x, y, method=method, **kwargs)
        
        if result is not None:
            baseline, corrected, error_metrics = result
            method_key = f"{method}_{len(self.baseline_methods_used)}" \
                if method in self.baseline_methods_used else method
            
            self.baselines[method_key] = baseline
            self.baseline_errors[method_key] = error_metrics
            self.baseline_methods_used.append(method_key)
            
            if len(self.baseline_methods_used) == 1 or kwargs.get('set_current', True):
                self.current_baseline = baseline
                self.baseline_wavenumbers = x
                self.baseline_corrected_intensity = corrected
                self.current_baseline_method = method_key
                self.u = self.baseline_wavenumbers
                self.v = self.baseline_corrected_intensity
                
                if kwargs.get('verbose', False):
                    print(f"  ✓ Created baseline-corrected data for spectrum {self.acquired}:")
                    print(f"    - baseline_wavenumbers (alias: u)")
                    print(f"    - baseline_corrected_intensity (alias: v)")
            
            return baseline, corrected, error_metrics
        else:
            print(f"Baseline method {method} failed for spectrum {self.acquired}")
            return None
    
    def get_baseline_summary(self):
        """Return summary of baseline corrections."""
        return get_baseline_summary(self.baselines, self.baseline_errors, 
                                   self.acquired, self.current_baseline_method)
    
    def plot_baselines(self):
        """Plot all computed baselines."""
        if hasattr(self, 'window_wavenumbers') and hasattr(self, 'window_intensity'):
            base_x = self.window_wavenumbers
            base_y = self.window_intensity
        else:
            base_x = self.x
            base_y = self.y
        
        plot_baselines(base_x, base_y, self.baselines, self.acquired)
    
    def continuous_wavelet_of_spectrum(self, wavelet='morl', scales=None):
        """Compute continuous wavelet transform."""
        if not hasattr(self, 'baseline_corrected_intensity'):
            raise AttributeError("Baseline-corrected data not found. Apply baseline correction first.")
        
        coefficients, frequencies = get_wavelet_transform(
            self.baseline_corrected_intensity, wavelet=wavelet, scales=scales
        )
        self.wavelet_coefficients = coefficients
        self.wavelet_frequencies = frequencies
        return coefficients, frequencies
    
    def plot_wavelet_coefficients(self):
        """Plot wavelet coefficients."""
        if not hasattr(self, 'wavelet_coefficients'):
            raise ValueError("Wavelet coefficients not computed. Call continuous_wavelet_of_spectrum() first.")
        
        plot_wavelet_coefficients(self.wavelet_coefficients, (self.x.min(), self.x.max()))
    
    def plot(self, normalized=True, color='r', size=1, **kwargs):
        """Plot the spectrum."""
        plot_spectrum(self.x, self.y, title=self.title, color=color, size=size)
    
    def plot_raw(self, color='b', size=1, **kwargs):
        """Plot raw spectrum."""
        plot_raw_spectrum(self.x, self.raw_y, title=f"Raw Spectrum: {self.title}", 
                         color=color, size=size)
    
    def reshape_range(self, range_limits, verbose=False):
        """Extract spectrum within specified range."""
        window_x, window_y, updated_baselines = reshape_range(
            self.x, self.y, range_limits, self.baselines
        )
        
        self.window_wavenumbers = window_x
        self.window_intensity = window_y
        self.p = self.window_wavenumbers
        self.q = self.window_intensity
        self.baselines = updated_baselines
        
        if verbose:
            print(f"  ✓ Created windowed data for spectrum {self.acquired}:")
            print(f"    - window_wavenumbers (alias: p) → x-axis in range {range_limits}")
            print(f"    - window_intensity (alias: q) → y-values in window")
            print(f"    - Data points: {len(self.window_wavenumbers)} (from original {len(self.x)})")
    
    def return_near_value(self, target_x):
        """Return nearest y value to target x."""
        return return_near_value(self.normdictcoords, target_x)
    
    def interpolate(self, final_size):
        """Interpolate spectrum to specified number of points."""
        if hasattr(self, 'window_wavenumbers') and hasattr(self, 'window_intensity'):
            expanded_x, expanded_y = interpolate_spectrum(
                self.window_wavenumbers, self.window_intensity, final_size
            )
            self.window_wavenumbers = expanded_x
            self.window_intensity = expanded_y
            self.p = expanded_x
            self.q = expanded_y
