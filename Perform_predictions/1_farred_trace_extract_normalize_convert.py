import numpy as np
from scipy.io import loadmat, savemat
from tkinter import filedialog, Tk
import os

from scipy.io.matlab.mio5_params import mat_struct

# ---  RECURSIVE STRUCTURE CONVERSION FUNCTIONS ---
def loadmat_recursive(filename):
    """Load a .mat file and fully convert mat_structs to dicts/lists"""
    def _todict(matobj):
        result = {}
        for field in matobj._fieldnames:
            val = getattr(matobj, field)
            result[field] = _convert(val)
        return result

    def _tolist(ndarray):
        return [_convert(x) for x in ndarray]

    def _convert(obj):
        if isinstance(obj, mat_struct):
            return _todict(obj)
        elif isinstance(obj, np.ndarray):
            return _tolist(obj)
        else:
            return obj

    data = loadmat(filename, struct_as_record=False, squeeze_me=True)
    return {k: _convert(v) for k, v in data.items() if not k.startswith('__')}

def extract_coloc_from_multiple_files():
    # Open file browser
    root = Tk()
    root.withdraw()
    file_paths = filedialog.askopenfilenames(
        title="Select .mat files containing gridData",
        filetypes=[("MAT files", "*.mat")]
    )

    all_traces = []
    all_spot_locations = []
    all_source_files = []

    for file_path in file_paths:
        print(f"Processing {os.path.basename(file_path)}")
        data = loadmat_recursive(file_path)
        
        grid_data = data['gridData']
        print(f"gridData keys: {list(grid_data.keys())}")
        example = list(grid_data.values())[0]
        print(f"Example entry type: {type(example)}")

        if 'gridData' not in data:
            print(f"  ⚠️  Skipping {file_path}: no gridData")
            continue

        grid_data = data['gridData']

        # Just handle the single image’s spot list
        spot_data = grid_data.get('FarRedSpotData', [])

        if isinstance(spot_data, dict):
            spot_data = [spot_data]

        for j, spot in enumerate(spot_data):
            if isinstance(spot, dict) and spot.get('colocGreen', 0) == 1:
                intensity = spot.get('intensityTrace', None)
                location = spot.get('spotLocation', None)
                if intensity is not None and location is not None:
                    intensity = (intensity - np.mean(intensity)) / (np.std(intensity) or 1)
                    all_traces.append(intensity)
                    all_spot_locations.append(location)
                    all_source_files.append(os.path.basename(file_path))

    if not all_traces:
        print("⚠️ No colocalized traces found in selected files.")
        return

    test_traces = np.vstack(all_traces)
    spot_locations = np.array(all_spot_locations)
    source_file = np.array(all_source_files, dtype=object).reshape(-1, 1)
    test_labels = np.zeros((test_traces.shape[0],), dtype=np.int64)  # optional dummy

    print(f"\n✅ Extracted {test_traces.shape[0]} colocalized traces across {len(file_paths)} files.")

    output_path = filedialog.asksaveasfilename(
        title="Save processed test data",
        defaultextension=".mat",
        filetypes=[("MAT files", "*.mat")],
        initialfile="20250505 - biological replicate 1 - farred traces.mat"
    )

    savemat(output_path, {
        'test_traces': test_traces,
        'test_labels': test_labels,
        'spotLocation': spot_locations,
        'sourceFile': source_file
    })

    print(f"✅ Saved to: {output_path}")

# Run it
if __name__ == "__main__":
    extract_coloc_from_multiple_files()
