import torch
import torch.nn as nn

class StepDetectionCNN(nn.Module):
    def __init__(self, input_length=1000, num_classes=6):
        super(StepDetectionCNN, self).__init__()
        
        # Calculate the output size after convolutions
        def calculate_output_size(input_length, pool_size=2):
            # Size after conv (with padding)
            conv_output = input_length
            # Size after pooling
            pool_output = (conv_output - pool_size) // pool_size + 1
            return pool_output
        
        # Apply the size calculation through each layer
        l1_size = calculate_output_size(input_length)
        l2_size = calculate_output_size(l1_size)
        l3_size = calculate_output_size(l2_size)
        
        # 1D Convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=251, padding=125),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(16, 32, kernel_size=125, padding=62),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(32, 64, kernel_size=63, padding=31),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * l3_size, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)  # predict up to num_classes-1 steps
        )
        
    def forward(self, x):
        # x shape: [batch_size, 1, trace_length]
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x