import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image

class ActionImageDataset(Dataset):
    def __init__(self, images_folder, actions_file, sequence_length=5, transform=None):
        self.data_dir = Path("cyber/models/world/dynamic/rssm")
        self.images_folder = images_folder
        self.actions = np.load(self.data_dir / actions_file)
        self.sequence_length = sequence_length
        self.transform = transform
        self.image_files = sorted(os.listdir(self.data_dir / images_folder))
        return
        # Ensure we have the right number of images and actions
        # assert len(self.image_files) == len(self.actions) + 1, \
        #     "Number of images must be N, and actions must be N-1."

    def __len__(self):
        # Total sequences possible
        return len(self.actions) - self.sequence_length + 1

    def __getitem__(self, idx):
        # Load 5 consecutive images
        image_sequence = []
        for i in range(idx, idx + self.sequence_length):
            img_path = self.data_dir / self.images_folder / self.image_files[i]
            img = Image.open(img_path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            image_sequence.append(img)

        # Stack images into a tensor
        image_sequence = torch.stack(image_sequence)

        # Get the corresponding actions
        action_sequence = torch.tensor(self.actions[idx:idx + self.sequence_length], dtype=torch.float32)

        return image_sequence, action_sequence