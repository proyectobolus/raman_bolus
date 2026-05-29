"""Batch processing operations."""
import numpy as np
import matplotlib.pyplot as plt


def reshape_all_spectra(all_raman, range_limits, verbose=False):
    """Apply range reshaping to all spectra."""
    for spectrum in all_raman.values():
        spectrum.reshape_range(range_limits, verbose=verbose)
    print(f"Reshaped all {len(all_raman)} spectra with window: {range_limits}")
    if verbose:
        print("  All spectra now have window_wavenumbers (p) and window_intensity (q)")


def interpolate_all_spectra(all_raman, final_size):
    """Apply interpolation to all spectra."""
    for spectrum in all_raman.values():
        spectrum.interpolate(final_size)
    print(f"Interpolated all {len(all_raman)} spectra with final_size: {final_size}")


def check_common_xrange(all_raman):
    """Check for common x-range across all spectra."""
    x_ranges = [(s.xmin, s.xmax) for s in all_raman.values()]
    common_xmin = max(x[0] for x in x_ranges)
    common_xmax = min(x[1] for x in x_ranges)
    
    if common_xmin >= common_xmax:
        raise ValueError("No common x-range found among spectra.")
    
    print(f"Common x-range found: {common_xmin} to {common_xmax} " +
          f"for batch of {len(all_raman)} spectra.")
    
    return common_xmin, common_xmax


def apply_baseline_to_batch(all_raman, methods=None, verbose=True, **kwargs):
    """Apply baseline correction to all spectra."""
    if methods is None:
        methods = ['poly3', 'modpoly', 'asls']
    
    if verbose:
        print(f"Applying baseline correction to {len(all_raman)} spectra using: {methods}")
    
    successful = 0
    results = {}
    
    for idx, spectrum in all_raman.items():
        if verbose:
            print(f"Processing spectrum {idx}: {spectrum.acquired}")
        
        spectrum_results = {}
        for method in methods:
            try:
                result = spectrum.baseline_correction(method, **kwargs)
                if result is not None:
                    baseline, corrected, errors = result
                    spectrum_results[method] = {
                        'success': True,
                        'rmse': errors.get('rmse', None),
                        'mae': errors.get('mae', None)
                    }
                    if verbose:
                        rmse = errors.get('rmse', 'N/A')
                        msg = f"  {method}: RMSE={rmse:.6f}" if isinstance(rmse, (int, float)) else f"  {method}: {rmse}"
                        print(msg)
                else:
                    spectrum_results[method] = {'success': False, 'error': 'Method failed'}
                    if verbose:
                        print(f"  {method}: Failed")
            except Exception as e:
                spectrum_results[method] = {'success': False, 'error': str(e)}
                if verbose:
                    print(f"  {method}: Error - {e}")
        
        results[idx] = spectrum_results
        if any(r['success'] for r in spectrum_results.values()):
            successful += 1
    
    if verbose:
        print(f"\nBaseline correction completed:")
        print(f"  Successful: {successful}/{len(all_raman)}")
        print(f"  Failed: {len(all_raman) - successful}")
    
    return results


def get_baseline_statistics(baseline_results):
    """Get statistics for baseline corrections."""
    if not baseline_results:
        return "No baseline corrections applied."
    
    stats = {}
    all_methods = set()
    
    for spectrum_results in baseline_results.values():
        all_methods.update(spectrum_results.keys())
    
    for method in all_methods:
        stats[method] = {
            'success_count': 0,
            'total_count': 0,
            'rmse_values': [],
            'mae_values': []
        }
    
    for spectrum_results in baseline_results.values():
        for method, result in spectrum_results.items():
            stats[method]['total_count'] += 1
            if result.get('success', False):
                stats[method]['success_count'] += 1
                if result.get('rmse') is not None:
                    stats[method]['rmse_values'].append(result['rmse'])
                if result.get('mae') is not None:
                    stats[method]['mae_values'].append(result['mae'])
    
    for method in all_methods:
        m_stats = stats[method]
        m_stats['success_rate'] = (m_stats['success_count'] / m_stats['total_count'] 
                                   if m_stats['total_count'] > 0 else 0)
        
        if m_stats['rmse_values']:
            m_stats['rmse_mean'] = np.mean(m_stats['rmse_values'])
            m_stats['rmse_std'] = np.std(m_stats['rmse_values'])
            m_stats['rmse_min'] = np.min(m_stats['rmse_values'])
            m_stats['rmse_max'] = np.max(m_stats['rmse_values'])
        
        if m_stats['mae_values']:
            m_stats['mae_mean'] = np.mean(m_stats['mae_values'])
            m_stats['mae_std'] = np.std(m_stats['mae_values'])
    
    return stats


def print_baseline_summary(baseline_results):
    """Print baseline correction summary."""
    stats = get_baseline_statistics(baseline_results)
    
    if isinstance(stats, str):
        print(stats)
        return
    
    print("Batch Baseline Correction Summary")
    print("=" * 60)
    
    for method, m_stats in stats.items():
        print(f"\nMethod: {method}")
        print(f"  Success Rate: {m_stats['success_rate']:.1%} " +
              f"({m_stats['success_count']}/{m_stats['total_count']})")
        
        if m_stats.get('rmse_mean') is not None:
            print(f"  RMSE - Mean: {m_stats['rmse_mean']:.6f}, Std: {m_stats['rmse_std']:.6f}")
            print(f"         Min: {m_stats['rmse_min']:.6f}, Max: {m_stats['rmse_max']:.6f}")
        
        if m_stats.get('mae_mean') is not None:
            print(f"  MAE - Mean: {m_stats['mae_mean']:.6f}, Std: {m_stats['mae_std']:.6f}")


def export_acquired_times(all_raman, batch_ids, output_file='selected.txt'):
    """Export acquired times to file."""
    existing_times = set()
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    existing_times.add(line)
        print(f"Loaded {len(existing_times)} existing times from {output_file}")
    except FileNotFoundError:
        print(f"File '{output_file}' not found. Will create new file.")
    except Exception as e:
        print(f"Warning: Error reading '{output_file}': {e}")
    
    times_to_append = []
    missing_ids = []
    
    for batch_id in batch_ids:
        if batch_id in all_raman:
            acquired_time = all_raman[batch_id].metadata['Acquired']
            if acquired_time in existing_times:
                print(f"  batch_id {batch_id}: Time already exists")
            else:
                print(f"  batch_id {batch_id}: Appending new time")
                times_to_append.append(acquired_time)
        else:
            missing_ids.append(batch_id)
    
    if missing_ids:
        print(f"Warning: These batch_ids not found: {missing_ids}")
    
    try:
        with open(output_file, 'a', encoding='utf-8') as f:
            for acquired_time in times_to_append:
                f.write(f"{acquired_time}\n")
        print(f"Appended {len(times_to_append)} times to {output_file}")
        return len(times_to_append)
    except Exception as e:
        print(f"Error writing to file '{output_file}': {e}")
        return 0


def plot_all_raw_spectra(all_raman, title='All Raw Spectra', alpha=0.6, linewidth=0.8):
    """Plot all raw (unnormalized) spectra in the batch.

    Parameters
    ----------
    all_raman : dict[int, RamanSpectrum]
        Mapping of batch IDs to spectrum instances
    title : str
        Figure title
    alpha : float
        Line transparency
    linewidth : float
        Line width for plotting
    """
    if not all_raman:
        print("No spectra to plot.")
        return

    plt.figure(figsize=(8, 5))
    for spectrum in all_raman.values():
        try:
            plt.plot(spectrum.x, spectrum.raw_y, '-', alpha=alpha, linewidth=linewidth)
        except Exception:
            pass

    plt.xlabel('Raman Shift (cm⁻¹)')
    plt.ylabel('Raw Intensity')
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_cleaned_spectra(
    all_raman,
    baseline_method='poly3',
    smooth=True,
    smooth_half_window=3,
    title='Baseline-Corrected Spectra',
    alpha=0.9,
    linewidth=0.8,
    verbose=False,
    **kwargs,
):
    """Perform baseline correction (default: poly3 smoothed) and plot cleaned spectra.

    Parameters
    ----------
    all_raman : dict[int, RamanSpectrum]
        Mapping of batch IDs to spectrum instances
    baseline_method : str
        Baseline method to apply (default 'poly3')
    smooth : bool
        Apply smoothing before baseline correction (default True)
    smooth_half_window : int
        Half-window for smoothing (default 3)
    title : str
        Figure title
    alpha : float
        Line transparency
    linewidth : float
        Line width for plotting
    verbose : bool
        Verbose logging during baseline correction
    **kwargs : dict
        Additional keyword args forwarded to `baseline_correction`
    """
    if not all_raman:
        print("No spectra to plot.")
        return

    plt.figure(figsize=(8, 5))

    for spectrum in all_raman.values():
        try:
            spectrum.baseline_correction(
                method=baseline_method,
                smooth=smooth,
                smooth_half_window=smooth_half_window,
                verbose=verbose,
                **kwargs,
            )

            x = getattr(spectrum, 'baseline_wavenumbers', spectrum.x)
            y = getattr(spectrum, 'baseline_corrected_intensity', None)
            if y is None:
                continue
            plt.plot(x, y, '-', alpha=alpha, linewidth=linewidth)
        except Exception:
            pass

    plt.xlabel('Raman Shift (cm⁻¹)')
    plt.ylabel('Baseline-Corrected Intensity')
    plt.title(title)
    plt.tight_layout()
    plt.show()
