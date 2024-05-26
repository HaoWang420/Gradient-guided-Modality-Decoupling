import torch

from torch.utils.data import Dataset

class SplitWrapper(Dataset):
    """Wrapper class for datasets to support distributed sampler for subset
    """

    def __init__(self, dataset: Dataset, indices) -> None:
        super().__init__()

        self.dataset = dataset
        self.indices = indices
    
    def __getitem__(self, index):
        return self.dataset[self.indices[index]]
    
    def __len__(self, ):
        return len(self.indices)