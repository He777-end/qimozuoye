import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class SequenceDataset(Dataset):
    """给 LSTM 用的 (X_seq, y) 数据集"""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X
        self.y = y

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        x = self.X[idx]  # (seq_len, feat_dim)
        y = self.y[idx]
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.float32)


def make_seq_dataloader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 64,
    shuffle: bool = True,
) -> DataLoader:
    ds = SequenceDataset(X, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
