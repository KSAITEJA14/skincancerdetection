import torch
import pathlib
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
import pandas as pd
from data_loading import ISICDataset
from training_pipeline import HybridModel, transform
from tqdm import tqdm
import pytz
from datetime import datetime
import torch.nn as nn
import torch.nn.init as init
import mlflow
import mlflow.pytorch
import yaml
import os
import signal
import sys

curr_dir = pathlib.Path(__file__)
home_dir = curr_dir.parent
params_path = home_dir.as_posix() + '/params.yaml'

with open(params_path, 'r') as stream:
    params = yaml.safe_load(stream)

# Parameters
experiment_name = params['experiment_name']
tracking_uri = params['tracking_uri']
batch_size = params['batch_size']
learning_rate = params['learning_rate']
num_epochs = params['num_epochs']
patience = params['patience']
train_csv_path = params['train_csv_path']
train_hdf5_path = params['train_hdf5_path']
model_path = params['model_path']
checkpoint_path = params['checkpoint_path']

print("Parameters loaded successfully")

# Set MLflow tracking URI and experiment
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment(experiment_name)

print("MLflow configured successfully")

# Load tabular data
train_df = pd.read_csv(train_csv_path)

# Create datasets and dataloaders
train_dataset = ISICDataset(train_df, train_hdf5_path, train_df['isic_id'], transform=transform, train=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Initialize the model
tabular_input_dim = train_df.shape[1] - 2  # Exclude 'target' and 'isic_id' columns
print(" Data load Completed, Start Model build")

model = HybridModel(tabular_input_dim=tabular_input_dim)


# Apply weight initialization
def initialize_weights(m):
    if isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight)
        if m.bias is not None:
            init.constant_(m.bias, 0)

model.apply(initialize_weights)

# Loss function and optimizer
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = Adam(model.parameters(), lr=learning_rate)

# Early stopping variables
best_loss = float('inf')
epochs_no_improve = 0

# Load checkpoint if exists
start_epoch = 0
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    best_loss = checkpoint['best_loss']
    print(f"Resuming training from epoch {start_epoch}")

# Function to save checkpoint
def save_checkpoint(epoch, model, optimizer, best_loss, checkpoint_path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_loss': best_loss
    }, checkpoint_path)
    print(f"Checkpoint saved at epoch {epoch}")

# Signal handler to save checkpoint on interrupt
def signal_handler(sig, frame):
    print("Training interrupted. Saving checkpoint...")
    save_checkpoint(epoch, model, optimizer, best_loss, checkpoint_path)
    sys.exit(0)

# Register signal handler
signal.signal(signal.SIGINT, signal_handler)

# Training loop
model.train()

with mlflow.start_run():
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("learning_rate", learning_rate)
    mlflow.log_param("num_epochs", num_epochs)
    mlflow.log_param("patience", patience)
    mlflow.log_param("train_csv_path", train_csv_path)
    mlflow.log_param("train_hdf5_path", train_hdf5_path)
    mlflow.log_param("model_path", model_path)

    for epoch in range(start_epoch, num_epochs):
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        for tabular_data, images, labels in progress_bar:
            labels = labels.float()  # Convert labels to float
            optimizer.zero_grad()
            outputs = model(tabular_data, images)
            
            if torch.isnan(outputs).any():
                raise ValueError("Outputs contain NaNs")
            
            loss = criterion(outputs, labels.unsqueeze(1))
            
            # Debugging: Print the loss value
            print(f"Loss: {loss.item()}")
            
            loss.backward()
            
            optimizer.step()
            running_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                "loss": running_loss / (progress_bar.n + 1),
                "lr": optimizer.param_groups[0]['lr']
            })
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for tabular_data, images, labels in val_loader:
                labels = labels.float()
                outputs = model(tabular_data, images)
                loss = criterion(outputs, labels.unsqueeze(1))
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        current_time_ny = datetime.now(pytz.timezone('America/New_York')).strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{current_time_ny}] Epoch {epoch+1}/{num_epochs}, Training Loss: {running_loss/len(train_loader)}, Validation Loss: {val_loss}, Learning Rate: {optimizer.param_groups[0]['lr']}")

        mlflow.log_metric("train_loss", running_loss/len(train_loader), step=epoch)
        mlflow.log_metric("val_loss", val_loss, step=epoch)

        # Save checkpoint at the end of each epoch
        save_checkpoint(epoch, model, optimizer, best_loss, checkpoint_path)

        # Early stopping check
        if val_loss < best_loss:
            best_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), model_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        model.train()

    # Final save of the trained model
    mlflow.pytorch.log_model(model, "model")
    torch.save(model.state_dict(), model_path)
