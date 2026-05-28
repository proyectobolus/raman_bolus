"""Baseline correction methods for Raman spectra."""
import numpy as np


def polynomial_baseline(x, y, degree=3):
    """Polynomial baseline fitting."""
    coef = np.polyfit(x, y, deg=degree)
    baseline = np.polyval(coef, x)
    corrected = y - baseline
    
    residuals = y - baseline
    rmse = np.sqrt(np.mean(residuals**2))
    mae = np.mean(np.abs(residuals))
    r_squared = 1 - np.sum(residuals**2) / np.sum((y - np.mean(y))**2)
    
    error_metrics = {
        'method': f'polynomial_degree_{degree}',
        'rmse': rmse,
        'mae': mae,
        'r_squared': r_squared,
        'max_residual': np.max(np.abs(residuals)),
        'parameters': {'degree': degree}
    }
    return baseline, corrected, error_metrics


def modpoly_baseline(x, y, degree=3, max_iter=100, tol=1e-3):
    """Modified polynomial baseline (iterative)."""
    baseline = np.polyval(np.polyfit(x, y, degree), x)
    
    for i in range(max_iter):
        mask = y < baseline
        if np.sum(mask) < degree + 1:
            break
            
        new_coef = np.polyfit(x[mask], y[mask], degree)
        new_baseline = np.polyval(new_coef, x)
        
        if np.max(np.abs(new_baseline - baseline)) < tol:
            break
        baseline = new_baseline
    
    corrected = y - baseline
    residuals = y - baseline
    rmse = np.sqrt(np.mean(residuals**2))
    mae = np.mean(np.abs(residuals))
    
    error_metrics = {
        'method': 'modpoly',
        'rmse': rmse,
        'mae': mae,
        'iterations': i + 1,
        'converged': i < max_iter - 1,
        'parameters': {'degree': degree, 'max_iter': max_iter, 'tolerance': tol}
    }
    return baseline, corrected, error_metrics


def asls_baseline(x, y, lam=1e7, p=0.02, max_iter=100):
    """Asymmetric Least Squares baseline."""
    from scipy import sparse
    from scipy.sparse.linalg import spsolve
    
    L = len(y)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L-2))
    w = np.ones(L)
    
    for i in range(max_iter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.T)
        baseline = spsolve(Z, w * y)
        w_new = p * (y > baseline) + (1 - p) * (y < baseline)
        
        if np.linalg.norm(w_new - w) / np.linalg.norm(w) < 1e-6:
            break
        w = w_new
    
    corrected = y - baseline
    residuals = y - baseline
    rmse = np.sqrt(np.mean(residuals**2))
    mae = np.mean(np.abs(residuals))
    
    error_metrics = {
        'method': 'asls',
        'rmse': rmse,
        'mae': mae,
        'iterations': i + 1,
        'parameters': {'lambda': lam, 'p': p, 'max_iter': max_iter}
    }
    return baseline, corrected, error_metrics


def snip_baseline(x, y, max_half_window=40, decreasing=True, smooth_half_window=3):
    """SNIP baseline."""
    spectrum = y.copy()
    
    if smooth_half_window > 0:
        from scipy.ndimage import uniform_filter1d
        spectrum = uniform_filter1d(spectrum, size=2*smooth_half_window+1)
    
    for i in range(max_half_window, 0, -1) if decreasing else range(1, max_half_window + 1):
        for j in range(i, len(spectrum) - i):
            spectrum[j] = min(spectrum[j], (spectrum[j-i] + spectrum[j+i]) / 2)
    
    baseline = spectrum
    corrected = y - baseline
    residuals = y - baseline
    rmse = np.sqrt(np.mean(residuals**2))
    mae = np.mean(np.abs(residuals))
    
    error_metrics = {
        'method': 'snip',
        'rmse': rmse,
        'mae': mae,
        'parameters': {
            'max_half_window': max_half_window,
            'decreasing': decreasing,
            'smooth_half_window': smooth_half_window
        }
    }
    return baseline, corrected, error_metrics


def mor_baseline(x, y, half_window=30):
    """Morphological baseline."""
    from scipy.ndimage import grey_closing, grey_opening
    
    ball_radius = half_window
    structuring_element = np.ones(2 * ball_radius + 1)
    baseline = grey_opening(y, structure=structuring_element)
    baseline = grey_closing(baseline, structure=structuring_element)
    corrected = y - baseline
    
    residuals = y - baseline
    rmse = np.sqrt(np.mean(residuals**2))
    mae = np.mean(np.abs(residuals))
    
    error_metrics = {
        'method': 'mor',
        'rmse': rmse,
        'mae': mae,
        'parameters': {'half_window': half_window}
    }
    return baseline, corrected, error_metrics


def zhang_baseline(x, y):
    """Zhang fit baseline."""
    spectrum = y.copy()
    for _ in range(10):
        spectrum = np.minimum(spectrum, np.convolve(spectrum, np.ones(5)/5, mode='same'))
    
    baseline = spectrum
    corrected = y - baseline
    residuals = y - baseline
    rmse = np.sqrt(np.mean(residuals**2))
    mae = np.mean(np.abs(residuals))
    
    error_metrics = {
        'method': 'zhang',
        'rmse': rmse,
        'mae': mae,
        'parameters': {}
    }
    return baseline, corrected, error_metrics
