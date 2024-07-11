import h5py
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import io

class ISICDataset(Dataset):
    def __init__(self, tabular_data, image_path, isic_ids, transform=None, train=True):
        self.tabular_data = tabular_data
        self.image_path = image_path
        self.isic_ids = isic_ids
        self.transform = transform
        self.train = train
        if train:
            self.labels = tabular_data['target'].values
            self.tabular_data = tabular_data.drop(columns=['target', 'isic_id'])
        else:
            self.tabular_data = tabular_data.drop(columns=['isic_id'])
        
    def __len__(self):
        return len(self.isic_ids)
    
    def __getitem__(self, idx):
        isic_id = self.isic_ids[idx]
        tabular_data = self.tabular_data.iloc[idx].values.astype(np.float32)
        
        with h5py.File(self.image_path, 'r') as images_file:
            img_data = images_file[isic_id][()]
            img = Image.open(io.BytesIO(img_data))
            if self.transform:
                img = self.transform(img)
            
            if self.train:
                label = self.labels[idx]
                return tabular_data, img, label
            else:
                return tabular_data, img
