import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import os
import tkinter as tk
from tkinter import filedialog
import pandas as pd

from model_39 import StepDetectionCNN

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
    # Parameters
    model_path = filedialog.askopenfilename(
        title="Select a .mat file with your model",
        filetypes=[("pth files", "*.pth")]
    )

    if not model_path:
        print("No model selected. Exiting.")
        return
    
    # Open file dialog to choose .mat file
    root = tk.Tk()
    root.withdraw()
    test_data_path = filedialog.askopenfilename(
        title="Select a .mat file with traces",
        filetypes=[("MAT files", "*.mat")]
    )

    if not test_data_path:
        print("No file selected. Exiting.")
        return
    
    # Load device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load test data
    try:
        test_data = loadmat(test_data_path)
        test_traces = test_data['test_traces']  # Update variable name based on your data structure
        
        has_labels = 'test_labels' in test_data
        if has_labels:
            test_labels = test_data['test_labels'].flatten()
            if np.unique(test_labels).size <= 1:
                print("⚠️ Only one label detected — treating as unlabeled.")
                has_labels = False

    except Exception as e:
        print(f"Error loading test data: {e}")
        return
    
    # Load model
    input_length = test_traces.shape[1]
    num_classes = 4  # Update based on your model's configuration
    model = load_model(model_path, input_length, num_classes, device)
    
    # Make predictions
    predictions, probabilities = analyze_batch(model, test_traces, device)
    
    # Evaluate if true labels are available
    if has_labels:
        accuracy = np.mean(predictions == test_labels.flatten())
        print(f"Test accuracy: {accuracy:.4f}")
        
        # Confusion matrix
        from sklearn.metrics import confusion_matrix, classification_report
        cm = confusion_matrix(test_labels.flatten(), predictions)
        print("Confusion Matrix:")
        print(cm)
        print("\nClassification Report:")
        print(classification_report(test_labels.flatten(), predictions))
    
    # Visualize some examples
    num_examples = min(5, len(test_traces))
    for i in range(num_examples):
        visualize_prediction(test_traces[i], predictions[i])
        if has_labels:
            print(f"True steps: {test_labels[i][0]}, Predicted steps: {predictions[i]}")
        else:
            print(f"Predicted steps: {predictions[i]}")

    if 'sourceFile' in test_data and 'spotLocation' in test_data:
        source_files = test_data['sourceFile'].flatten()
        spot_location = test_data['spotLocation'].flatten()

        import csv
        output_csv = os.path.splitext(os.path.basename(test_data_path))[0] + "_predictions.csv"
        
        with open(output_csv, mode='w', newline='') as file:
            writer = csv.writer(file)
            header = ['sourceFile', 'spotLocation', 'predicted_steps'] + [f'prob_class_{i}' for i in range(probabilities.shape[1])]
            writer.writerow(header)
            for i in range(len(predictions)):
                row = [source_files[i], spot_location[i], predictions[i]] + probabilities[i].tolist()
                writer.writerow(row)
        
        print(f"✅ Saved predictions to: {output_csv}")
    else:
        print("⚠️ Could not save predictions — sourceFile/spotLocation missing from .mat.")

    # Build DataFrame from predictions
    print(f"len(source_files): {len(source_files)}")
    print(f"len(spot_location): {len(spot_location)}")
    print(f"len(predictions): {len(predictions)}")

    if 'sourceFile' in test_data and 'spotLocation' in test_data:
        source_files = test_data['sourceFile'].flatten()
        raw_spot_location = test_data['spotLocation']
        spot_location = [f"{loc[0]}_{loc[1]}" if isinstance(loc, (np.ndarray, list, tuple)) else str(loc) for loc in raw_spot_location]

        source_files = [str(s[0]) if isinstance(s, np.ndarray) else str(s) for s in source_files]

        df = pd.DataFrame({
            'sourceFile': source_files,
            'spotLocation': spot_location,
            'predicted_steps': predictions
        })

        # Filter for predictions > 1
        df_filtered = df[df['predicted_steps'] > 0]

        # Save to CSV
        filtered_csv = os.path.splitext(os.path.basename(test_data_path))[0] + "_predicted_steps_gt0.csv"
        df_filtered.to_csv(filtered_csv, index=False)
        print(f"✅ Saved filtered predictions (steps > 0) to: {filtered_csv}")

        # Save full prediction table
        output_csv = os.path.splitext(os.path.basename(test_data_path))[0] + "_final_model_predictions.csv"
        df.to_csv(output_csv, index=False)
        print(f"✅ Saved predictions to: {output_csv}")

        # Generate summary counts per file per class
        summary = df.groupby(['sourceFile', 'predicted_steps']).size().unstack(fill_value=0)
        summary.columns = [f'class_{i}' for i in summary.columns]

        summary_csv = os.path.splitext(os.path.basename(test_data_path))[0] + "_final_model_summary.csv"
        summary.to_csv(summary_csv)
        print(f"✅ Saved summary to: {summary_csv}")
        
        # Create confidence threshold summary tables and write to Excel
        confidence_thresholds = [0.0, 0.6, 0.7, 0.8, 0.9, 0.95]
        predicted_confidences = probabilities.max(axis=1)

        # Prepare Excel writer
        excel_filename = os.path.splitext(os.path.basename(test_data_path))[0] + "_final_model_summaries.xlsx"
        with pd.ExcelWriter(excel_filename, engine='xlsxwriter') as writer:
            for thresh in confidence_thresholds:
                mask = predicted_confidences >= thresh
                filtered_preds = predictions[mask]
                filtered_sources = test_data['sourceFile'].flatten()[mask]
                filtered_traces = np.array(spot_location)[mask]

                filtered_sources = [str(s[0]) if isinstance(s, np.ndarray) else str(s) for s in filtered_sources]
                df_thresh = pd.DataFrame({
                    'sourceFile': filtered_sources,
                    'spotLocation': filtered_traces,
                    'predicted_steps': filtered_preds
                })

                summary = df_thresh.groupby(['sourceFile', 'predicted_steps']).size().unstack(fill_value=0)
                summary.columns = [f'class_{i}' for i in summary.columns]

                # Compute total counts for class 1–3
                for class_id in [1, 2, 3]:
                    if f'class_{class_id}' not in summary.columns:
                        summary[f'class_{class_id}'] = 0  # Add missing class column if not present

                summary['total_1_2_3'] = summary[['class_1', 'class_2', 'class_3']].sum(axis=1)

                # Calculate percentage of each class among 1–3
                for class_id in [1, 2, 3]:
                    summary[f'percent_class_{class_id}'] = (
                        summary[f'class_{class_id}'] / summary['total_1_2_3'].replace(0, np.nan)
                    ) * 100

                sheet_name = f'thresh_{int(thresh*100)}'
                summary.to_excel(writer, sheet_name=sheet_name, float_format="%.2f")

                print(f"✅ Added sheet for ≥{int(thresh*100)}% confidence")

        print(f"✅ All summaries with totals and percentages saved to Excel: {excel_filename}")

if __name__ == "__main__":
    main()
