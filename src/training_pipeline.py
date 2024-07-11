import h5py
import numpy as np
from torchvision import transforms
from torchvision.models import resnet50
from PIL import Image
import torch
import torch.nn as nn

def extract_image_from_hdf5(hdf5_path, output_path):
    with h5py.File(hdf5_path, 'r') as images_file:
        
        for key in images_file.keys():
            img = np.array(images_file[key])
            img = Image.fromarray(img)
            yield img
            
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class TabularModel(nn.Module):
    def __init__(self, input_dim, embedding_dim=64):
        super(TabularModel , self).__init__()
        self.fc1 = nn.Linear(input_dim, embedding_dim)
        self.fc2 = nn.Linear(embedding_dim, 32)
        self.fc3 = nn.Linear(32, 16)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return x
    
class ImageModel(nn.Module):
    def __init__(self, num_classes=64):
        super(ImageModel, self).__init__()
        self.base_model = resnet50(weights='ResNet50_Weights.DEFAULT')
        self.base_model = nn.Sequential(*list(self.base_model.children())[:-1])
        self.fc1 = nn.Linear(2048, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 16)
        
    def forward(self, x):
        x= self.base_model(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        return x

class HybridModel(nn.Module):
    def __init__(self, tabular_input_dim):
        super(HybridModel, self).__init__()
        self.image_model = ImageModel(num_classes=64)
        self.tabular_model = TabularModel(input_dim=tabular_input_dim, embedding_dim=64)
        self.fc1 = nn.Linear(16 + 16, 16)
        self.fc2 = nn.Linear(16, 1)
        
    def forward(self, tabular_data, image_data):
        tab_out = self.tabular_model(tabular_data)
        img_out= self.image_model(image_data)
        hybrid_in = torch.cat((tab_out, img_out), dim=1)
        x = torch.relu(self.fc1(hybrid_in))
        x = torch.sigmoid(self.fc2(x))  # Ensure sigmoid activation for output
        return x
