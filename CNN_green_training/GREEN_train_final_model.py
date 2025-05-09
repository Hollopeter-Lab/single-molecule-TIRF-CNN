import argparse
import tkinter as tk
from tkinter import filedialog
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import pandas as pd
from datetime import datetime
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.utils.class_weight import compute_class_weight

from model_32 import StepDetectionCNN
from data_loading import load_matlab_data

def create_dataloaders(traces, labels, batch_size):
    tensor_traces = torch.tensor(traces, dtype=torch.float32).unsqueeze(1)
    tensor_labels = torch.tensor(labels, dtype=torch.long)
    dataset = torch.utils.data.TensorDataset(tensor_traces, tensor_labels)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader

def train_one_fold(model, loader, criterion, optimizer, device, num_epochs):
    history = {'train_loss': [], 'train_acc': []}
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        for traces, labels in loader:
            traces, labels = traces.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(traces)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * traces.size(0)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
        history['train_loss'].append(train_loss / total)
        history['train_acc'].append(correct / total)
    return model, history

def main():
    parser = argparse.ArgumentParser(description="Train final model on all data")
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()

    root = tk.Tk(); root.withdraw()
    file_path = filedialog.askopenfilename(title="Select your training dataset (.mat)", filetypes=[("MAT files", "*.mat")])
    if not file_path:
        raise SystemExit("No file selected.")

    traces, labels = load_matlab_data(file_path)
    labels = np.squeeze(labels)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    save_dir = f"Final_trained_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(save_dir, exist_ok=True)

    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    loader = create_dataloaders(traces, labels, args.batch)

    model = StepDetectionCNN(input_length=traces.shape[1], num_classes=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

    model, history = train_one_fold(model, loader, criterion, optimizer, device, num_epochs=args.epochs)

    # Evaluate
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for traces_batch, labels_batch in loader:
            traces_batch = traces_batch.to(device)
            outputs = model(traces_batch)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels_batch.numpy())
            y_pred.extend(preds.cpu().numpy())

    report = classification_report(y_true, y_pred, output_dict=True, target_names=["Reject", "Monomer"])
    cm = confusion_matrix(y_true, y_pred)

    excel_path = os.path.join(save_dir, 'final_training_summary.xlsx')
    with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
        pd.DataFrame(history).to_excel(writer, sheet_name='Training_History', index=False)
        pd.DataFrame(cm, index=["True_Reject", "True_Monomer"],
                        columns=["Pred_Reject", "Pred_Monomer"]).to_excel(writer, sheet_name='Confusion_Matrix')
        pd.DataFrame(report).transpose().to_excel(writer, sheet_name='Classification_Report')
    torch.save(model.state_dict(), os.path.join(save_dir, 'final_model_green_selected.pth'))
    print("✅ Final model training complete and saved.")
  
if __name__ == "__main__":
    main()
