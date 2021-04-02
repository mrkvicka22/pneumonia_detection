import os
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
import numpy as np


class PneumoniaDataset(Dataset):
    def __init__(self, path="", transforms=None):
        # stuff
        self.path = path
        self.transforms = transforms

    def __getitem__(self, index):
        # stuff
        ...
        img = Image.open(os.path.join(self.path, f"{index:08d}", "RTG.jpeg"))
        label_file = os.path.join(self.path, f"{index:08d}", "diagnose.txt")
        with open(label_file, "r") as lf:
            label = int(lf.readline())
        if self.transforms is not None:
            img = self.transforms(img)
        # If the transform variable is not empty
        # then it applies the operations in the transforms with the order that it is created.
        return (img, label)

    def __len__(self):
        return len(os.listdir(self.path))  # of how many data(images?) you have


if __name__ == '__main__':
    # Define transforms (1)
    transformations = transforms.Compose([transforms.Grayscale(),transforms.Resize((128,128)), transforms.ToTensor(),transforms.Normalize(mean=0.48814950165, std=0.24329058187339847)])
    # Call the dataset

    custom_dataset = PneumoniaDataset(path=r"C:\Users\Matej\PycharmProjects\pneumonia_detection\data_indexer", transforms=transformations)

    class_weights = [1 / 1493, 1 / 2780, 1 / 1583]
    dataloader = DataLoader(custom_dataset, batch_size=1,shuffle=False, num_workers=4)