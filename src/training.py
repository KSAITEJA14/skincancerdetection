import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
import pandas as pd
from data_loading import ISICDataset
from training_pipeline import HybridModel, transform

# Ensure you have set your paths correctly
train_csv_path = 'data/processed/train.csv'
train_hdf5_path = 'data/raw/isic-2024-challenge/train-image.hdf5'
model_path = 'Models/hybrid_model.pth'
batch_size = 32

# Load tabular data
train_df = pd.read_csv(train_csv_path)

# Create dataset and dataloader
train_dataset = ISICDataset(train_df, train_hdf5_path, train_df['isic_id'], transform=transform, train=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Initialize the model
tabular_input_dim = train_df.shape[1] - 2  # Exclude 'target' and 'isic_id' columns
model = HybridModel(tabular_input_dim=tabular_input_dim)

# Loss function and optimizer
criterion = torch.nn.BCELoss()
optimizer = Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
model.train()
for epoch in range(num_epochs):
    running_loss = 0.0
    for tabular_data, images, labels in train_loader:
        # Check for NaNs in the input data
        if torch.isnan(tabular_data).any():
            raise ValueError("Input tabular data contains NaNs")
        if torch.isnan(images).any():
            raise ValueError("Input images data contains NaNs")
        if torch.isnan(labels).any():
            raise ValueError("Input labels data contains NaNs")

        labels = labels.float()  # Convert labels to float
        optimizer.zero_grad()
        outputs = model(tabular_data, images)
        
        # Debugging: Print the range of outputs
        print(f"Outputs range: min={outputs.min().item()}, max={outputs.max().item()}")
        
        if torch.isnan(outputs).any():
            raise ValueError("Outputs contain NaNs")
        
        if not torch.all((outputs >= 0) & (outputs <= 1)):
            raise ValueError(f"Outputs out of range: {outputs}")
        
        loss = criterion(outputs, labels.unsqueeze(1))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")

# Save the trained model
torch.save(model.state_dict(), model_path)
