"""Baseline correction orchestration."""
from . import baseline_methods


def apply_baseline_correction(x, y, method='poly3', **kwargs):
    """Apply baseline correction with specified method."""
    try:
        if method.startswith('poly'):
            degree = int(method.replace('poly', ''))
            return baseline_methods.polynomial_baseline(x, y, degree=degree)
        elif method == 'modpoly':
            return baseline_methods.modpoly_baseline(x, y, **kwargs)
        elif method == 'asls':
            return baseline_methods.asls_baseline(x, y, **kwargs)
        elif method == 'snip':
            return baseline_methods.snip_baseline(x, y, **kwargs)
        elif method == 'mor':
            return baseline_methods.mor_baseline(x, y, **kwargs)
        elif method == 'zhang':
            return baseline_methods.zhang_baseline(x, y, **kwargs)
        else:
            print(f"Unknown baseline method: {method}")
            return None
    except Exception as e:
        print(f"Error in baseline method {method}: {e}")
        return None


def get_baseline_summary(baselines, baseline_errors, acquired='', current_method=''):
    """Get summary of baseline corrections."""
    if not baselines:
        return "No baseline corrections applied."
    
    summary = f"Baseline Corrections{f' for {acquired}' if acquired else ''}:\n"
    summary += "=" * 50 + "\n"
    
    for method, baseline in baselines.items():
        errors = baseline_errors.get(method, {})
        summary += f"Method: {method}\n"
        
        rmse_val = errors.get('rmse', 'N/A')
        if isinstance(rmse_val, (int, float)):
            summary += f"  RMSE: {rmse_val:.6f}\n"
        else:
            summary += f"  RMSE: {rmse_val}\n"
        
        mae_val = errors.get('mae', 'N/A')
        if isinstance(mae_val, (int, float)):
            summary += f"  MAE: {mae_val:.6f}\n"
        else:
            summary += f"  MAE: {mae_val}\n"
        
        r2_val = errors.get('r_squared', 'N/A')
        if isinstance(r2_val, (int, float)):
            summary += f"  R²: {r2_val:.6f}\n"
        else:
            summary += f"  R²: {r2_val}\n"
        
        if 'iterations' in errors:
            summary += f"  Iterations: {errors['iterations']}\n"
        summary += "\n"
    
    if current_method:
        summary += f"Current baseline: {current_method}\n"
    
    return summary
