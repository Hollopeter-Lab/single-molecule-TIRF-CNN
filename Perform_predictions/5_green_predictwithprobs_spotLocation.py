import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import os
import tkinter as tk
from tkinter import filedialog
import pandas as pd
from pandas import ExcelWriter
from model_11 import StepDetectionCNN

def load_model(model_path, input_length, num_classes, device='cuda'):
    """
    Load trained model
    
    Args:
        model_path (str): Path to the saved model
        input_length (int): Length of input traces
        num_classes (int): Number of classes
        device (str): Device to load model on
        
    Returns:
        model: Loaded model
    """
    model = StepDetectionCNN(input_length=input_length, num_classes=num_classes)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()
    return model

def predict_steps(model, trace, device='cuda'):
    """
    Predict number of steps in a single trace
    
    Args:
        model: Neural network model
        trace (numpy.ndarray): Single intensity trace
        device (str): Device to use for prediction
        
    Returns:
        int: Predicted number of steps
    """
    # Preprocess trace
    trace_tensor = torch.FloatTensor(trace).unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions [1, 1, trace_length]
    trace_tensor = trace_tensor.to(device)
    
    # Make prediction
    with torch.no_grad():
        output = model(trace_tensor)
        _, predicted = torch.max(output, 1)
    
    return predicted.item()

def visualize_prediction(trace, predicted_steps):
    """
    Visualize trace and predicted steps
    
    Args:
        trace (numpy.ndarray): Intensity trace
        predicted_steps (int): Predicted number of steps
    """
    plt.figure(figsize=(10, 6))
    plt.plot(trace)
    plt.title(f'Predicted Steps: {predicted_steps}')
    plt.xlabel('Frame')
    plt.ylabel('Intensity')
    plt.grid(True, alpha=0.3)
    plt.show()

def analyze_batch(model, traces, device='cuda'):
    """
    Analyze a batch of traces
    
    Args:
        model: Neural network model
        traces (numpy.ndarray): Batch of intensity traces
        device (str): Device to use for prediction
        
    Returns:
        numpy.ndarray: Predicted number of steps for each trace
    """
    # Convert traces to tensor
    traces_tensor = torch.FloatTensor(traces).unsqueeze(1)  # Add channel dimension [batch_size, 1, trace_length]
    traces_tensor = traces_tensor.to(device)
    
    # Make predictions
    with torch.no_grad():
        outputs = model(traces_tensor)
        probs = torch.softmax(outputs, dim=1)  # Apply softmax to get probabilities
        _, predicted = torch.max(probs, 1)
        return predicted.cpu().numpy(), probs.cpu().numpy()

def main():
    # Initialize Tkinter
    root = tk.Tk()
    root.withdraw()
    print("Hi.")

    # Select model file
    model_path = filedialog.askopenfilename(
        title="Select a .pth model file",
        filetypes=[("PyTorch model", "*.pth")]
    )
    if not model_path:
        print("No model selected. Exiting.")
        return

    # Select CSV file
    csv_path = filedialog.askopenfilename(
        title="Select matched_green_spots.csv",
        filetypes=[("CSV files", "*.csv")]
    )
    if not csv_path:
        print("No CSV file selected. Exiting.")
        return

    # Load and filter data
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")
    df = df[df['green_trace'].notnull()].reset_index(drop=True)
    print(f"{len(df)} rows have non-null green_trace")

    # Parse traces from strings to lists
    green_traces = df['green_trace'].apply(eval).tolist()
    green_traces = np.array(green_traces)

    # Setup model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    input_length = green_traces.shape[1]
    num_classes = 2
    model = load_model(model_path, input_length, num_classes, device)

    # Predict
    predictions, probabilities = analyze_batch(model, green_traces, device)
    df['green_predicted_steps'] = predictions
    for i in range(probabilities.shape[1]):
        df[f'green_prob_class_{i}'] = probabilities[:, i]

    # Build summary
    summary_df = df.groupby(['green_predicted_steps']).size().reset_index(name='count')

    # Prepare Excel output path
    excel_output_path = filedialog.asksaveasfilename(
        title="Save Excel file with predictions and summary",
        defaultextension=".xlsx",
        filetypes=[("Excel files", "*.xlsx")],
        initialfile="20250505 - biological replicate 1 - green predictions matched with farred predictions.xlsx"
    )
    if not excel_output_path:
        print("No Excel save location selected. Exiting.")
        return
    
    # Cross-tab: green vs far-red predicted steps
    cross_tab = pd.crosstab(df['green_predicted_steps'], df['predicted_steps'], rownames=['green_predicted_steps'], colnames=['farRed_predicted_steps'], dropna=False)

    # Ensure all expected columns are present (0–3)
    for col in range(4):
        if col not in cross_tab.columns:
            cross_tab[col] = 0
    cross_tab = cross_tab[[0, 1, 2, 3]]  # Ensure consistent order

    # Write both sheets to Excel
    with ExcelWriter(excel_output_path, engine='xlsxwriter') as writer:
        # Summary sheet
        summary_df.to_excel(writer, sheet_name='summary', index=False)

        # Detail sheet with specific columns
        detail_columns = [
            'farRed_sourceFile',
            'farRed_spotLocation',
            'predicted_steps',
            'green_predicted_steps',
            'green_spotLocation',
            'green_sourceFile',
            'green_trace'
        ]
        df[detail_columns].to_excel(writer, sheet_name='detailed_predictions', index=False)
        
        # Green step counts per green_sourceFile
        step_counts = df.groupby(['green_sourceFile', 'green_predicted_steps']).size().unstack(fill_value=0)

        # Ensure all class columns (0-3) are present
        for col in range(4):
            if col not in step_counts.columns:
                step_counts[col] = 0

        # Optional: rename columns for clarity
        step_counts = step_counts.rename(columns={
            0: 'class_0',
            1: 'class_1',
            2: 'class_2',
            3: 'class_3'
        })
        step_counts['total'] = step_counts[['class_0', 'class_1', 'class_2', 'class_3']].sum(axis=1)

        for class_id in [1, 2, 3]:
            step_counts[f'percent_class_{class_id}'] = (
                step_counts[f'class_{class_id}'] / step_counts['total'].replace(0, np.nan)
            ) * 100

        step_counts = step_counts.reset_index()
        step_counts.to_excel(writer, sheet_name='green_step_counts_per_file', index=False)

        cross_tab.to_excel(writer, sheet_name='green_vs_farRed_matrix')
        
        # Filter to green class 0 only
        mono_df = df[df['green_predicted_steps'] == 1]

        # Count far-red step predictions per farRed_sourceFile
        mono_counts = mono_df.groupby(['farRed_sourceFile', 'predicted_steps']).size().unstack(fill_value=0)

        # Ensure all far-red classes (1–3) are present
        for col in range(4):
            if col not in mono_counts.columns:
                mono_counts[col] = 0
        mono_counts = mono_counts[[0, 1, 2, 3]]  # Consistent order

        # Save to new sheet
        mono_counts.to_excel(writer, sheet_name='farRed_dist_if_green1')

    print(f"✅ Excel file with summary and detailed predictions saved to: {excel_output_path}")

    # Save extended CSV
    output_csv = filedialog.asksaveasfilename(
        title="Save CSV of green trace predictions",
        defaultextension=".csv",
        filetypes=[("CSV files", "*.csv")],
        initialfile="20250505 - biological replicate 1 - green predictions matched with farred predictions.csv"
    )
    if not output_csv:
        print("No save location selected. Exiting.")
        return
    
    df.to_csv(output_csv, index=False)
    print(f"✅ Saved full predictions to: {output_csv}")

    # Summary
    summary = df.groupby(['green_predicted_steps']).size().rename('count').reset_index()
    print("\n✅ Prediction Summary:")
    print(summary)

    summary_csv = os.path.splitext(csv_path)[0] + "20250505 - biological replicate 1 - green predictions matched with farred predictions.csv"
    summary.to_csv(summary_csv, index=False)
    print(f"✅ Saved summary to: {summary_csv}")

    # Optional: preview a few traces
    for i in range(min(5, len(green_traces))):
        visualize_prediction(green_traces[i], predictions[i])

if __name__ == "__main__":
    main()