import argparse
import tkinter as tk
from tkinter import filedialog
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import time
from data_loading import load_matlab_data, create_dataloaders
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import pandas as pd
from datetime import datetime
from sklearn.utils.class_weight import compute_class_weight
import importlib

# --------------------------------------------------------------
# Argument parsing + optional file‑chooser for the .mat dataset
# --------------------------------------------------------------
parser = argparse.ArgumentParser(description="Train photobleach step CNN with hyperparameter sweep")
parser.add_argument("--data", nargs='*', help="Paths to one or more .mat files with labelled traces")
parser.add_argument("--epochs", type=int, default=50)
args = parser.parse_args()

# If no data files were passed through --data, open a file dialog
if not args.data:
    root = tk.Tk(); root.withdraw()
    data_paths = filedialog.askopenfilenames(
        title="Select one or more .mat files containing labelled traces",
        filetypes=[("MAT‑files", "*.mat")]
    )
    if not data_paths:
        raise SystemExit("No data file selected — exiting.")
    data_paths = list(data_paths)  # tkinter returns a tuple; make it a list
else:
    data_paths = args.data  # use command-line specified files


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50, device='cuda', timestamp=None, save_dir=None):
    """
    Train the neural network
    
    Args:
        model (nn.Module): Neural network model
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        criterion: Loss function
        optimizer: Optimizer
        num_epochs (int): Number of training epochs
        device (str): Device to use for training ('cuda' or 'cpu')
        
    Returns:
        model: Trained model
        dict: Training history
    """
    model = model.to(device)
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    best_val_loss = float('inf')
    patience = 5  # <-- you can adjust this
    epochs_no_improve = 0
    early_stop = False

    start_time = time.time()
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for traces, labels in train_bar:
            traces, labels = traces.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(traces)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            train_loss += loss.item() * traces.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            train_bar.set_postfix({
                'loss': f"{loss.item():.4f}", 
                'acc': f"{train_correct/train_total:.4f}"
            })
        
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Valid]")
            for traces, labels in val_bar:
                traces, labels = traces.to(device), labels.to(device)
                
                outputs = model(traces)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * traces.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                val_bar.set_postfix({
                    'loss': f"{loss.item():.4f}", 
                    'acc': f"{val_correct/val_total:.4f}"
                })
        
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_correct / val_total
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save best model and early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = os.path.join(save_dir, f'best_step_detection_model_{timestamp}.pth')
            torch.save(model.state_dict(), model_path)
            print("Model saved!")
            epochs_no_improve = 0  # reset counter
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                early_stop = True
        if early_stop:
            break
    
    time_elapsed = time.time() - start_time
    print(f"Training completed in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    
    # Load best model
    model.load_state_dict(torch.load(model_path))
    
    return model, history

def plot_training_history(history, timestamp=None, save_dir=None):
    """
    Plot training and validation loss and accuracy
    
    Args:
        history (dict): Training history dictionary
    """
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'training_history_{timestamp}.png'))

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    learning_rates = [1e-3, 5e-4, 1e-4]
    batch_sizes = [16, 32, 64]
    num_epochs = args.epochs

    for model_num in range(1, 40): 
        model_module_name = f"model_{model_num}"
        model_module = importlib.import_module(model_module_name)
        StepDetectionCNN = model_module.StepDetectionCNN

        for learning_rate in learning_rates:
            for batch_size in batch_sizes:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                model_label = f"model_{model_num}"
                save_dir = f"GREEN_CNN_{model_label}_{timestamp}_lr{learning_rate}_bs{batch_size}"
                os.makedirs(save_dir, exist_ok=True)
                
                print(f"🚀 {model_label} | LR={learning_rate}, BS={batch_size}")

                print(f"📁 Saving results to folder: {save_dir}")
                
                # Set device
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                print(f"Using device: {device}")

                all_traces = []
                all_labels = []

                for path in data_paths:
                    traces, labels = load_matlab_data(path)
                    if traces is not None and labels is not None:
                        all_traces.append(traces)
                        all_labels.append(labels)
                    else:
                        print(f"⚠️ Warning: Failed to load {path}")

                # Concatenate all loaded data
                if not all_traces:
                    print("No valid training data loaded. Exiting...")
                    return

                traces = np.vstack(all_traces)
                labels = np.hstack(all_labels)

                print(f"Loaded {len(traces)} traces with shape {traces.shape}")
                
                # Create data loaders
                train_loader, val_loader = create_dataloaders(traces, labels, batch_size)
                
                # Grab all training labels (from the dataset)
                train_labels_array = np.array([
                    train_loader.dataset[i][1].item() for i in range(len(train_loader.dataset))
                ])

                # Compute class weights
                unique_classes = np.unique(train_labels_array)
                weights = compute_class_weight(class_weight='balanced', classes=unique_classes, y=train_labels_array)
                weights = torch.tensor(weights, dtype=torch.float).to(device)

                # Create model
                input_length = traces.shape[1]  # Length of each trace
                num_classes = int(labels.max()) + 1  # Number of classes (0 to max number of steps)
                model = StepDetectionCNN(input_length=input_length, num_classes=num_classes)
                
                # Define loss function and optimizer
                criterion = nn.CrossEntropyLoss(weight=weights)
                optimizer = optim.Adam(model.parameters(), lr=learning_rate)
                
                # Train model
                model, history = train_model(
                    model, train_loader, val_loader, criterion, optimizer, 
                    num_epochs=num_epochs, device=device, timestamp=timestamp, save_dir=save_dir
                )
                
                # Plot training history
                plot_training_history(history, timestamp, save_dir=save_dir)
                
                # Evaluate on validation set (or a separate test set if available)
                model.eval()
                all_preds = []
                all_labels = []

                with torch.no_grad():
                    for traces_batch, labels_batch in val_loader:  
                        traces_batch = traces_batch.to(device)
                        outputs = model(traces_batch)
                        _, preds = torch.max(outputs, 1)
                        all_preds.extend(preds.cpu().numpy())
                        all_labels.extend(labels_batch.numpy())

                # Confusion matrix
                cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
                labels_names = ['Reject', 'Monomeric']

                # Print raw matrix
                print("📊 Confusion Matrix:")
                print(pd.DataFrame(cm, index=labels_names, columns=labels_names))

                # Optional: Fancy plot
                plt.figure(figsize=(6, 5))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels_names, yticklabels=labels_names)
                plt.xlabel("Predicted")
                plt.ylabel("True")
                plt.title("Confusion Matrix (Validation Set)")
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f"confusion_matrix_{timestamp}.png"))

                # Classification report
                print("\n🧾 Classification Report:")
                print(classification_report(all_labels, all_preds, target_names=labels_names, digits=4))

                # Print detailed F1 score report
                print("Class weights:", dict(zip(unique_classes, weights.cpu().numpy().round(2))))

                print("Training completed successfully!")

                # Prepare data for Excel output
                report_dict = classification_report(all_labels, all_preds, target_names=labels_names, output_dict=True)
                df_report = pd.DataFrame(report_dict).transpose()

                # Overall Accuracy
                overall_accuracy = np.mean(np.array(all_preds) == np.array(all_labels))

                # Class Weights (as a DataFrame)
                df_class_weights = pd.DataFrame({
                    'class': list(unique_classes),
                    'class_weight': weights.cpu().numpy()
                })

                # Confusion Matrix (as a DataFrame)
                df_cm = pd.DataFrame(cm, index=[f"True_{i}" for i in labels_names], columns=[f"Pred_{i}" for i in labels_names])

                # Create an Excel writer
                excel_filename = os.path.join(save_dir, f"training_summary_{timestamp}.xlsx")
                with pd.ExcelWriter(excel_filename, engine='xlsxwriter') as writer:
                    # Sheet 1: Classification Report
                    df_report.to_excel(writer, sheet_name='Classification_Report')
                    
                    # Sheet 2: Confusion Matrix
                    df_cm.to_excel(writer, sheet_name='Confusion_Matrix')
                    
                    # Sheet 3: Class Weights
                    df_class_weights.to_excel(writer, sheet_name='Class_Weights')
                    
                    # Sheet 4: Overall metrics
                    df_overall = pd.DataFrame({'Metric': ['Overall Accuracy'], 'Value': [overall_accuracy]})
                    df_overall.to_excel(writer, sheet_name='Overall', index=False)

                print(f"✅ Saved training summary to: {excel_filename}")

                # --- Append best validation accuracy and F1 score to folder name ---
                best_val_acc = max(history['val_acc']) * 100
                best_val_acc_str = f"{best_val_acc:.1f}%"

                # Extract macro average F1 score from classification report
                macro_f1 = report_dict['macro avg']['f1-score'] * 100
                macro_f1_str = f"{macro_f1:.1f}%"

                # Safe formatting for filenames
                acc_str = best_val_acc_str.replace('.', 'p')
                f1_str = macro_f1_str.replace('.', 'p')

                plt.close('all')
                time.sleep(1)  # Give time for OS to release file handles

                # Rename folder with both metrics
                new_save_dir = f"{save_dir}_valacc{acc_str}_f1{f1_str}"

                # Only rename if different
                if new_save_dir != save_dir:
                    os.rename(save_dir, new_save_dir)
                    print(f"📂 Renamed output folder to: {new_save_dir}")

if __name__ == "__main__":
    main()