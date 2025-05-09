import argparse
import tkinter as tk
from tkinter import filedialog
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import pandas as pd
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score, accuracy_score
from sklearn.utils.class_weight import compute_class_weight

from model_32 import StepDetectionCNN
from data_loading import load_matlab_data


def create_dataloaders(traces, labels, batch_size):
    tensor_traces = torch.tensor(traces, dtype=torch.float32).unsqueeze(1)
    tensor_labels = torch.tensor(labels, dtype=torch.long)
    dataset = torch.utils.data.TensorDataset(tensor_traces, tensor_labels)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader


def train_one_fold(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=50):
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_loss = float('inf')
    patience = 5
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for traces, labels in train_loader:
            traces, labels = traces.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(traces)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * traces.size(0)
            _, preds = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (preds == labels).sum().item()

        train_loss /= len(train_loader.dataset)
        train_acc = train_correct / train_total

        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for traces, labels in val_loader:
                traces, labels = traces.to(device), labels.to(device)
                outputs = model(traces)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * traces.size(0)
                _, preds = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (preds == labels).sum().item()

        val_loss /= len(val_loader.dataset)
        val_acc = val_correct / val_total

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            break

    return model, history, epoch+1


def main():
    parser = argparse.ArgumentParser(description="Train with K-Fold Cross Validation")
    parser.add_argument('--folds', type=int, default=5, help='Number of K folds')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()

    root = tk.Tk(); root.withdraw()
    file_path = filedialog.askopenfilename(title="Select your training dataset (.mat)", filetypes=[("MAT files", "*.mat")])
    if not file_path:
        raise SystemExit("No file selected.")

    traces, labels = load_matlab_data(file_path)
    labels = np.squeeze(labels)
  
    save_dir = f"K_fold_cross_val_model_32_20250504_133123_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(save_dir, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)
    confusion_matrices = []
    all_folds_summary = []

    excel_filename = os.path.join(save_dir, 'model_32_20250504_133123_training_summary.xlsx')
    with pd.ExcelWriter(excel_filename, engine='xlsxwriter') as writer:
        for fold, (train_idx, val_idx) in enumerate(skf.split(traces, labels)):
            print(f"Training fold {fold+1}/{args.folds}")
            X_train, y_train = traces[train_idx], labels[train_idx]
            X_val, y_val = traces[val_idx], labels[val_idx]

            class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
            class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

            train_loader = create_dataloaders(X_train, y_train, args.batch)
            val_loader = create_dataloaders(X_val, y_val, args.batch)

            model = StepDetectionCNN(input_length=traces.shape[1], num_classes=2).to(device)
            optimizer = optim.Adam(model.parameters(), lr=args.lr)
            criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

            model, history, num_epochs_ran = train_one_fold(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=args.epochs)

            model.eval()
            y_true, y_pred = [], []
            with torch.no_grad():
                for traces_batch, labels_batch in val_loader:
                    traces_batch = traces_batch.to(device)
                    outputs = model(traces_batch)
                    _, preds = torch.max(outputs, 1)
                    y_true.extend(labels_batch.numpy())
                    y_pred.extend(preds.cpu().numpy())

            cm = confusion_matrix(y_true, y_pred)
            confusion_matrices.append(cm)

            df_cm = pd.DataFrame(cm, index=["True_Reject", "True_Monomer"], columns=["Pred_Reject", "Pred_Monomer"])
            df_cm.to_excel(writer, sheet_name=f'Confusion_Fold_{fold+1}')

            df_history = pd.DataFrame(history)
            df_history.to_excel(writer, sheet_name=f'History_Fold_{fold+1}', index=False)

            report = classification_report(y_true, y_pred, output_dict=True, target_names=["Reject", "Monomer"])
            metrics_summary = {
            'Fold': fold+1,
            'Training Loss (final)': history['train_loss'][-1],
            'Training Accuracy (final)': history['train_acc'][-1],
            'Validation Loss (final)': history['val_loss'][-1],
            'Validation Accuracy (final)': history['val_acc'][-1],
            'Epochs Run': num_epochs_ran,
            'Precision Reject': report['Reject']['precision'],
            'Recall Reject': report['Reject']['recall'],
            'F1 Score Reject': report['Reject']['f1-score'],
            'Precision Monomer': report['Monomer']['precision'],
            'Recall Monomer': report['Monomer']['recall'],
            'F1 Score Monomer': report['Monomer']['f1-score'],
            'Macro Avg F1': report['macro avg']['f1-score'],
            'Macro Avg Accuracy': accuracy_score(y_true, y_pred)
            }
            
            all_folds_summary.append(metrics_summary)

            df_metrics_summary = pd.DataFrame(metrics_summary, index=[0])
            df_metrics_summary.to_excel(writer, sheet_name=f'Metrics_Fold_{fold+1}', index=False)

            classes_present = np.unique(y_train)
            df_class_weights = pd.DataFrame({
                'Class': classes_present,
                'Class Label': ['Reject' if c == 0 else 'Monomer' for c in classes_present],
                'Class Weight': class_weights
            })
            df_class_weights.to_excel(writer, sheet_name=f'ClassWeights_Fold_{fold+1}', index=False)

            torch.save(model.state_dict(), os.path.join(save_dir, f'fold_{fold+1}_model.pth'))

        mean_cm = np.mean(confusion_matrices, axis=0)
        df_mean_cm = pd.DataFrame(mean_cm, index=["True_Reject", "True_Monomer"], columns=["Pred_Reject", "Pred_Monomer"])
        df_mean_cm.to_excel(writer, sheet_name='Mean_Confusion_Matrix')
        df_all_folds_summary = pd.DataFrame(all_folds_summary)
        df_all_folds_summary.to_excel(writer, sheet_name='All_Folds_Summary', index=False)

    plt.figure(figsize=(8,6))
    sns.heatmap(df_mean_cm, annot=True, fmt='.2f', cmap='Blues')
    plt.title('Mean Confusion Matrix Across Folds')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'mean_confusion_matrix.png'))
    plt.show()


if __name__ == "__main__":
    main()
