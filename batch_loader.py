"""Batch file loading and spectrum discovery."""
import os
from datetime import datetime
from ..raman_spectrum import RamanSpectrum


def load_all_spectra(root=None, avoided_dirs=None, file_suffix='.txt', 
                    first_line_prefix="#Acq. time (s)=", excluded_times=None,
                    roots=None, included_times=None):
    """Load all spectrum files from one or multiple directory trees.

    Parameters
    ----------
    root : str, optional
        Single root directory to walk (ignored if `roots` provided)
    roots : list[str], optional
        List of root directories to walk; enables multi-directory search
    avoided_dirs : set[str] | list[str], optional
        Directory names to skip while walking
    file_suffix : str, optional
        File suffix to include (default '.txt')
    first_line_prefix : str, optional
        Prefix required in first line to validate spectrum file
    excluded_times : set[str] | list[str], optional
        Acquisition times to exclude from loading
    included_times : set[str] | list[str], optional
        Acquisition times to include; if provided, only spectra with these times are loaded

    Returns
    -------
    dict[int, RamanSpectrum]
        Mapping of sequential IDs to loaded spectra
    """
    avoided = set(avoided_dirs or [])
    excluded = set(excluded_times or [])
    included = set(included_times or []) if included_times else None
    all_raman = {}
    counter = 0

    # Build list of roots to search
    search_roots = []
    if roots:
        search_roots.extend(list(roots))
    elif root:
        search_roots.append(root)
    else:
        search_roots.append('.')

    for base_root in search_roots:
        for root_dir, dirs, files in os.walk(base_root):
            dirs[:] = [d for d in dirs if d not in avoided and not d.startswith('.')]

            for fname in files:
                if not fname.endswith(file_suffix):
                    continue

                filepath = os.path.join(root_dir, fname)
                if not _file_matches(filepath, first_line_prefix):
                    continue

                try:
                    spectrum = RamanSpectrum(filepath)
                    acquired_time = spectrum.metadata.get('Acquired')

                    if acquired_time in excluded:
                        print(f"Skipping spectrum with excluded time: {acquired_time}")
                        continue

                    # If included_times is provided, only include spectra with matching times
                    if included is not None and acquired_time not in included:
                        print(f"Skipping spectrum not in include list: {acquired_time}")
                        continue

                    all_raman[counter] = spectrum
                    counter += 1
                except Exception as e:
                    print(f"Error loading {filepath}: {e}")

    return all_raman


def _file_matches(filepath, first_line_prefix):
    """Check if file matches loading criteria."""
    try:
        with open(filepath, 'r', encoding='iso-8859-1') as f:
            first_line = f.readline()
        return first_line.startswith(first_line_prefix)
    except Exception:
        return False


def load_excluded_times(filepath):
    """Load excluded acquisition times from file."""
    excluded = set()
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    excluded.add(line)
        print(f"Loaded {len(excluded)} excluded times from {filepath}")
    except FileNotFoundError:
        print(f"Warning: File '{filepath}' not found")
    except Exception as e:
        print(f"Error reading file '{filepath}': {e}")
    
    return excluded


def load_included_times(filepath):
    """Load included acquisition times from file."""
    included = set()
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    included.add(line)
        print(f"Loaded {len(included)} included times from {filepath}")
    except FileNotFoundError:
        print(f"Warning: File '{filepath}' not found")
    except Exception as e:
        print(f"Error reading file '{filepath}': {e}")
    
    return included


def deduplicate_and_sort_spectra(all_raman):
    """Remove duplicates and assign batch IDs in chronological order."""
    seen_times = set()
    cleaned = {}
    
    for key, spectrum in all_raman.items():
        acquired_time = spectrum.metadata['Acquired']
        if acquired_time not in seen_times:
            seen_times.add(acquired_time)
            cleaned[key] = spectrum
    
    sorted_spectra = sorted(
        cleaned.items(),
        key=lambda item: _parse_acquisition_time(item[1])
    )
    
    sorted_raman = {}
    acquired_times = []
    
    for new_id, (old_key, spectrum) in enumerate(sorted_spectra):
        spectrum.batch_id = new_id
        sorted_raman[new_id] = spectrum
        acquired_times.append(spectrum.metadata['Acquired'])
    
    return sorted_raman, acquired_times


def _parse_acquisition_time(spectrum):
    """Parse acquisition timestamp for sorting."""
    try:
        acquired_str = spectrum.metadata['Acquired']
        return datetime.strptime(acquired_str, '%d.%m.%Y %H:%M:%S')
    except (ValueError, KeyError):
        try:
            return datetime.strptime(acquired_str, '%Y-%m-%d %H:%M:%S')
        except (ValueError, KeyError):
            print(f"Warning: Could not parse acquisition time '{acquired_str}'")
            return datetime.min
