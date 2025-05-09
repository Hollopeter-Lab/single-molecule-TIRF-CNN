import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from torch.utils.data import Dataset, DataLoader
import os
from sklearn.model_selection import train_test_split

class PhotobleachingDataset(Dataset):
    def __init__(self, traces, labels, transform=None):
        """
        Args:
            traces (numpy.ndarray): Intensity time traces
            labels (numpy.ndarray): Number of steps in each trace
            transform (callable, optional): Optional transform to apply to the traces
        """
        self.traces = traces
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.traces)
    
    def __getitem__(self, idx):
        trace = self.traces[idx]
        label = self.labels[idx]
        
        # Convert to torch tensors
        trace = torch.FloatTensor(trace).unsqueeze(0)  # Add channel dimension [1, trace_length]
        label = torch.tensor(label, dtype=torch.long).squeeze()
        
        if self.transform:
            trace = self.transform(trace)
            
        return trace, label

def load_matlab_data(file_path):
    """
    Load and process MATLAB data file containing intensity traces
    
    Args:
        file_path (str): Path to MATLAB .mat file
        
    Returns:
        tuple: (traces, labels) arrays
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    try:
        # Load the MATLAB file
        data = loadmat(file_path)
        
        # Extract traces and labels (adjust variable names based on your MATLAB structure)
        # This assumes your .mat file has 'traces' and 'labels' variables
        traces = data.get('traces', None)
        labels = data.get('labels', None)
        
        # Convert NaNs labels to 0 and cast to int
        labels = np.nan_to_num(labels, nan=0).astype(np.int64)

        if traces is None or labels is None:
            print(f"Available keys in the .mat file: {data.keys()}")
            raise KeyError("Could not find 'traces' or 'labels' in the MATLAB file")
        
        return traces, labels
    
    except Exception as e:
        print(f"Error loading MATLAB file: {e}")
        return None, None

def create_dataloaders(traces, labels, batch_size=32, train_ratio=0.8):
    """
    Create train and validation DataLoaders
    
    Args:
        traces (numpy.ndarray): Intensity time traces
        labels (numpy.ndarray): Number of steps in each trace
        batch_size (int): Batch size for training
        train_ratio (float): Portion of data to use for training
        
    Returns:
        tuple: (train_loader, val_loader) DataLoader objects
    """
    # Flatten labels if necessary
    if labels.ndim > 1:
        labels = labels.squeeze()

    # Remove classes with fewer than 2 instances
    unique, counts = np.unique(labels, return_counts=True)
    valid_classes = unique[counts >= 2]

    # Create a mask to keep only valid samples
    mask = np.isin(labels, valid_classes)
    traces = traces[mask]
    labels = labels[mask]

    # Stratified split
    X_train, X_val, y_train, y_val = train_test_split(
        traces, labels,
        test_size=(1 - train_ratio),
        stratify=labels,
        random_state=42
    )
    
    print("Train label distribution:", dict(zip(*np.unique(y_train, return_counts=True))))
    print("Val label distribution:", dict(zip(*np.unique(y_val, return_counts=True))))

    train_dataset = PhotobleachingDataset(X_train, y_train)
    val_dataset = PhotobleachingDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader