import pandas as pd
import scipy.io
import numpy as np
import tkinter as tk
from tkinter import filedialog
import os

# Hide root window
root = tk.Tk()
root.withdraw()

# Select CSV file
csv_path = filedialog.askopenfilename(title="Select CSV with FarRed spot predictions", filetypes=[("CSV files", "*.csv")])
if not csv_path:
    raise ValueError("No CSV file selected.")

# Select MAT file
mat_path = filedialog.askopenfilename(title="Select .mat file containing GreenSpotData", filetypes=[("MAT files", "*.mat")])
if not mat_path:
    raise ValueError("No .mat file selected.")

# Load and parse CSV
df = pd.read_csv(csv_path)
df['x'] = df['spotLocation'].apply(lambda s: int(s.split('_')[0]))
df['y'] = df['spotLocation'].apply(lambda s: int(s.split('_')[1]))

# Load MAT file
mat = scipy.io.loadmat(mat_path, struct_as_record=False, squeeze_me=True)
green_spots = mat['spotLocation']
green_traces = mat['test_traces']
green_sources = [str(s[0]) if isinstance(s, np.ndarray) else str(s) for s in mat['sourceFile']]


# Match FarRed (x, y) to Green spots within 2-pixel tolerance
matched_green_traces = []

for _, row in df.iterrows():
    fx, fy = row['x'], row['y']
    match_found = False
    
    for i, (gx, gy) in enumerate(green_spots):
        green_file = str(green_sources[i]).strip()
        farred_file = str(row['sourceFile']).strip()

        if green_file == row['sourceFile'].strip() and abs(int(gx) - int(fx)) <= 3 and abs(int(gy) - int(fy)) <= 3:
            matched_green_traces.append({
                'farRed_spotLocation': f"{fx}_{fy}",
                'predicted_steps': row['predicted_steps'],
                'green_spotLocation': f"{gx}_{gy}",
                'green_trace': green_traces[i].tolist(),
                'farRed_sourceFile': row['sourceFile'],
                'green_sourceFile': green_file,
            })
            match_found = True
            break

    if not match_found:
        print(f"No match for FarRed: ({fx}, {fy}) in file: {farred_file}")
        matched_green_traces.append({
            'farRed_sourceFile': row['sourceFile'],
            'predicted_steps': row['predicted_steps'],
            'farRed_spotLocation': f"{fx}_{fy}",
            'green_spotLocation': None,
            'green_sourceFile': None,
            'green_trace': None,
        })

# Save results
out_path = filedialog.asksaveasfilename(
    title="Save matched green spot output as...",
    defaultextension=".csv",
    filetypes=[("CSV files", "*.csv")],
    initialfile="20250505 - biological replicate 1 - farred matched to green traces.csv"
)
if not out_path:
    raise ValueError("No output path selected.")

matched_df = pd.DataFrame(matched_green_traces)
matched_df.to_csv(out_path, index=False)
print(f"✅ Saved to {out_path}")
