"""Data loading and processing for Raman spectra."""
import numpy as np


def load_spectrum_file(filepath):
    """Load spectrum data from file."""
    with open(filepath, 'r', encoding='iso-8859-1') as f:
        lines = f.readlines()
    
    metadata = {}
    for line in lines:
        if line.startswith("#"):
            key, value = line.strip().split("=")
            metadata[key[1:]] = value.replace('\t', '')
        else:
            break
    
    data = np.loadtxt(lines[len(metadata):])
    x = data[:, 0]
    y = data[:, 1]
    return metadata, x, y, data


def process_metadata(metadata):
    """Extract and process metadata."""
    acquired = metadata.get('Acquired', '')
    date_acquired = acquired.split(' ')[0] if acquired else ''
    time_acquired = acquired.split(' ')[1] if acquired else ''
    title = metadata.get('Title', '').replace(' ', '_')
    
    range_str = metadata.get('Range (cm-¹)', '0...0')
    ranges = range_str.split('...')
    xmin = int(ranges[0])
    xmax = int(ranges[1])
    
    return {
        'acquired': acquired,
        'date_acquired': date_acquired,
        'time_acquired': time_acquired,
        'title': f"{title} {acquired}",
        'xmin': xmin,
        'xmax': xmax,
        'total_range': xmax - xmin,
    }


def normalize_spectrum(y):
    """Normalize spectrum by maximum value."""
    return y / np.max(y)


def create_coordinate_dicts(data, y_normalized):
    """Create coordinate dictionaries."""
    dictcoords = {point[0]: point[1] for point in data}
    normdictcoords = {point[0]: point[1] / np.max(y_normalized) for point in data}
    return dictcoords, normdictcoords
