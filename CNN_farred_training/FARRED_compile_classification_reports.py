import os
import pandas as pd
import tkinter as tk
from tkinter import filedialog

# --- GUI folder selection ---
root = tk.Tk()
root.withdraw()
root_folder = filedialog.askdirectory(title="Select the parent folder containing all model subfolders")
if not root_folder:
    raise SystemExit("No folder selected. Exiting.")

def extract_and_pivot_classification_reports(root_folder):
    data_rows = []

    for dirpath, _, filenames in os.walk(root_folder):
        for file in filenames:
            if file.startswith("training_summary_") and file.endswith(".xlsx"):
                full_path = os.path.join(dirpath, file)
                try:
                    df = pd.read_excel(full_path, sheet_name='Classification_Report', index_col=0)
                    
                    # Extract last 3 folders for traceability
                    parts = os.path.normpath(full_path).split(os.sep)[-4:-1]
                    model_id = "/".join(parts)

                    # Flatten the classification report (multi-index to wide format)
                    df_flat = df.T.unstack().to_frame().T
                    df_flat.columns = [f"{idx[1]}_{idx[0]}" for idx in df_flat.columns]
                    df_flat.insert(0, 'Model', model_id)

                    data_rows.append(df_flat)
                except Exception as e:
                    print(f"⚠️ Failed to load {full_path}: {e}")

    if data_rows:
        final_df = pd.concat(data_rows, ignore_index=True)
        output_path = os.path.join(root_folder, "compiled_model_metrics_wide.xlsx")
        final_df.to_excel(output_path, index=False)
        print(f"✅ Saved combined metrics to: {output_path}")
    else:
        print("❌ No valid classification report sheets found.")

# Run the function with the selected folder
extract_and_pivot_classification_reports(root_folder)
