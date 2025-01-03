import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch

from utils.author_100recent import get_100_recent_posts
import utils.dataset_processors as dataset_processors

class DatasetLoader(Dataset):
    
    def __init__(self, input, attention, labels):

        self._input = input
        self._attention = attention
        self._labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self._labels) 
    
    def __getitem__(self, idx):
        return (self._input[idx], self._attention[idx], self._labels[idx])


class MyMapDataset(Dataset):
    def __init__(self, dataset, tokenizer, token_length, DEVICE, mode):

        print(f"Dataset is {dataset}")
        author_ids, input_ids, targets = None, None, None

        if dataset == "essays":
            datafile = "data/essays/essays.csv"
            author_ids, input_ids, targets = dataset_processors.essays_embeddings(
                datafile, tokenizer, token_length, mode
            )

        elif dataset == 'kaggle':
            datafile = "data/kaggle/kaggle.csv"
            author_ids, input_ids, targets = dataset_processors.kaggle_embeddings(
                datafile, tokenizer, token_length
            )

        elif dataset == "pandora":
            author_ids, input_ids, targets = dataset_processors.pandora_embeddings(
                datafile, tokenizer, token_length
            )

        author_ids = torch.from_numpy(np.array(author_ids)).long().to(DEVICE)
        input_ids = torch.from_numpy(np.array(input_ids)).long().to(DEVICE)
        targets = torch.from_numpy(np.array(targets))

        if dataset == "pandora":
            targets = targets.float().to(DEVICE)
        else:
            targets = targets.long().to(DEVICE)

        self.author_ids = author_ids
        self.input_ids = input_ids
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return (self.author_ids[idx], self.input_ids[idx], self.targets[idx])
    

class FineTunedDataset(Dataset):
    def __init__(self, cls_features, labels, samples):
        self._features = cls_features
        self._labels = labels
        self._samples = samples
    
    def __len__(self):
        return len(self._labels)
    
    def __getitem__(self, idx):
        return (self._features[idx], self._labels[idx], self._samples[idx])

