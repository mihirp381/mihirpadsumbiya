import numpy as np
import torch
from torch import nn

L2 = 0.01


class CrackCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 128, 3)
        self.conv2 = nn.Conv2d(128, 128, 3)
        self.pool = nn.MaxPool2d(4)
        self.fc1 = nn.Linear(6272, 128)
        self.fc2 = nn.Linear(128, 1)
        for m in (self.conv1, self.conv2, self.fc1, self.fc2):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = torch.relu(self.fc1(torch.flatten(x, 1)))
        return self.fc2(x).squeeze(1)

    def penalty(self):
        return L2 * (self.fc1.weight.pow(2).sum() + self.fc1.bias.pow(2).sum())


def to_input(u8, device):
    return torch.from_numpy(np.ascontiguousarray(u8)).to(device).permute(0, 3, 1, 2).float().div(255.0)


def split_indices(y, seed=42):
    from sklearn.model_selection import train_test_split
    idx = np.arange(len(y))
    rest, test = train_test_split(idx, test_size=4000, stratify=y, random_state=seed)
    train, val = train_test_split(rest, test_size=3600, stratify=y[rest], random_state=seed)
    return np.sort(train), np.sort(val), np.sort(test)


@torch.no_grad()
def predict(model, X, idx, device, bs=500):
    model.eval()
    out = []
    for i in range(0, len(idx), bs):
        out.append(torch.sigmoid(model(to_input(X[idx[i:i + bs]], device))).cpu().numpy())
    return np.concatenate(out).astype(np.float64)
