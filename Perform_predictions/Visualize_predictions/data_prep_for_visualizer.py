import pandas as pd
import scipy.io
import os
from tkinter import Tk, filedialog

def load_mat_file(mat_filepath):
    data = scipy.io.loadmat(mat_filepath)
    source_files = [str(s[0]).strip() if isinstance(s[0], str) else s[0][0] for s in data['sourceFile']]
    spot_locations = data['spotLocation']
    farred_traces = data['test_traces']
    return source_files, spot_locations, farred_traces

def match_farred_trace(row, source_files, spot_locations, farred_traces):
    try:
        x_str, y_str = row['farRed_spotLocation'].split('_')
        x, y = int(x_str), int(y_str)
        filename = row['farRed_sourceFile'].strip()
        for idx, (fx, fy) in enumerate(spot_locations):
            if int(fx) == x and int(fy) == y and source_files[idx] == filename:
                return farred_traces[idx].tolist()
        return None
    except Exception:
        return None

def generate_enriched_file(excel_path, mat_path, output_path):
    df = pd.read_csv(excel_path)
    source_files, spot_locations, farred_traces = load_mat_file(mat_path)
    df['farred_trace'] = df.apply(lambda row: match_farred_trace(row, source_files, spot_locations, farred_traces), axis=1)
    df.to_csv(output_path, index=False)
    print(f"Enriched file saved to: {output_path}")

if __name__ == "__main__":
    Tk().withdraw()  # Hide the root window
    
    print("Select the Excel file containing far-red predictions matched to colocalized green traces...")
    excel_file = filedialog.askopenfilename(title="Select Excel File containing far-red predictions matched to colocalized green traces", filetypes=[("CSV Files", "*.csv")])

    print("Select the MATLAB file with far-red traces...")
    mat_file = filedialog.askopenfilename(title="Select .mat File with far-red traces", filetypes=[("MATLAB Files", "*.mat")])

    if excel_file and mat_file:
        output_file = filedialog.asksaveasfilename(defaultextension=".csv", title="Save Enriched File As", filetypes=[("CSV Files", "*.csv")])
        if output_file:
            generate_enriched_file(excel_file, mat_file, output_file)
        else:
            print("No output file selected. Operation cancelled.")
    else:
        print("File selection cancelled.")
