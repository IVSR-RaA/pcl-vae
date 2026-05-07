import os
import numpy as np
import torch
from torch.utils.data import Dataset
import glob

class RangeImageDataset(Dataset):
    
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.file_paths = self._get_all_paths(root_dir)

    def _get_all_paths(self, root_dir):
        # Recursively get all file paths from the root directory
        return glob.glob(os.path.join(root_dir, '**', '*.npy'), recursive=True)
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        range_path = self.file_paths[idx]
        range_np = np.load(range_path)

        # Match the previous torchvision ToTensor() output for HxW numpy arrays:
        # produce a float tensor with shape (1, H, W).
        range_tensor = torch.from_numpy(range_np).float().unsqueeze(0)

        return range_tensor
