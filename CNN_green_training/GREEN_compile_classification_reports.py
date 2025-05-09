import os
import pandas as pd
from glob import glob
import tkinter as tk
from tkinter import filedialog

# --- GUI folder selection ---
root = tk.Tk()
root.withdraw()
parent_dir = filedialog.askdirectory(title="Select the parent folder containing all model subfolders")
if not parent_dir:
    raise SystemExit("No folder selected. Exiting.")

# Output file
output_excel = os.path.join(parent_dir, "compiled_classification_reports.xlsx")

# Store all model summaries here
summary_rows = []

# Traverse subfolders
for root, dirs, files in os.walk(parent_dir):
    for file in files:
        if file.startswith("training_summary") and file.endswith(".xlsx"):
            file_path = os.path.join(root, file)
            model_name = os.path.basename(os.path.dirname(file_path))

            try:
                # Load classification report sheet
                df = pd.read_excel(file_path, sheet_name="Classification_Report", index_col=0)

                # Flatten to a single row, prefixing each column with the class/metric
                flattened = df.stack().to_frame().T
                flattened.columns = [f"{idx}_{metric}" for idx, metric in flattened.columns]

                # Add model name for traceability
                flattened.insert(0, "Model", model_name)

                summary_rows.append(flattened)

            except Exception as e:
                print(f"⚠️ Failed to load {file_path}: {e}")

# Combine all rows into a DataFrame
if summary_rows:
    summary_df = pd.concat(summary_rows, ignore_index=True)
    summary_df.to_excel(output_excel, index=False)
    print(f"✅ Summary saved to: {output_excel}")
else:
    print("❌ No valid classification report sheets found.")
