import torch
from torch.utils.data import DataLoader
import pandas as pd
from data_loading import ISICDataset
from training_pipeline import HybridModel, transform

# Ensure you have set your paths correctly
test_csv_path = 'data/processed/test.csv'
test_hdf5_path = 'data/raw/isic-2024-challenge/test-image.hdf5'
model_path = 'Models/hybrid_model.pth'
batch_size = 32

# Load tabular data
test_df = pd.read_csv(test_csv_path)

# Create dataset and dataloader
test_dataset = ISICDataset(test_df, test_hdf5_path, test_df['isic_id'], transform=transform, train=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize the model
tabular_input_dim = test_df.shape[1] - 1  # Exclude 'isic_id' column
model = HybridModel(tabular_input_dim=tabular_input_dim)

# Load the trained model
model.load_state_dict(torch.load(model_path))
model.eval()

# Evaluation on test data
predictions = []
with torch.no_grad():
    for tabular_data, images in test_loader:
        outputs = model(tabular_data, images)
        predictions.extend(outputs.squeeze().tolist())

# Create a DataFrame for submission
submission_df = pd.DataFrame({
    'isic_id': test_df['isic_id'],
    'target': predictions
})
submission_df.to_csv('data/processed/submission.csv', index=False)
