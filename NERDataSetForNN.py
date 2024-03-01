import torch
from torch.utils.data import Dataset

class NERDataSetForNN(Dataset):
    """
    A dataset class for NER tasks that interfaces with PyTorch's Dataset class.
    """
    def __init__(self, vectors, labels):
        """
        Initializes the dataset with preprocessed data from an NERDataSet instance.
        """
        self.vectors = torch.tensor(vectors, dtype=torch.float)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Retrieves the vector and label for a given index.
        """
        return self.vectors[idx], self.labels[idx]
