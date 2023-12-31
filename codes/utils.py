import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset
from copy import deepcopy

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ParityDataset(Dataset):
    def __init__(
        self,
        n_samples,
        n_elems,
        model='rnn',
        noise=True,
    ):
        self.n_samples = n_samples
        self.n_elems = n_elems

        self.model = model
        self.noise = noise

        self.X, self.Y = [], []
        self.noisy_X, self.noisy_Y = [], []
        if self.n_samples > 0:
            self.build_dataset()

    def __len__(self):
        return self.n_samples

    def generate_data(self):
        while True:
            x = torch.zeros((self.n_elems,))
            n_non_zero = torch.randint(
                1, self.n_elems, (1,)
            ).item()

            if self.noise:
                x[:n_non_zero] = torch.randint(0, 2, (n_non_zero,)) * 2 - 1
            else:
                x[:n_non_zero] = 1.0

            x = x[torch.randperm(self.n_elems)]

            y = (x == 1.0).sum() % 2

            return x, y.item()


    def build_dataset(self):
        for _ in range(self.n_samples):
            x, y = self.generate_data()
            self.X.append(x)
            self.Y.append(y)

        self.Y = torch.Tensor(self.Y).float()
        if self.model == 'rnn':
            self.X = torch.stack(self.X).float().unsqueeze(dim=-1)
        elif self.model == 'mlp':
            self.X = torch.stack(self.X).float()
        elif self.model == 'cnn':
            self.X = torch.stack(self.X).to(torch.int64) + 1
            self.X = F.one_hot(self.X, num_classes=3).float()

        perm_index = torch.randperm(self.X.size()[0])
        self.X = self.X[perm_index]
        self.Y = self.Y[perm_index]

        self.noisy_X = deepcopy(self.X)
        self.noisy_Y = deepcopy(self.Y)

    def add_noisy_label(self, noisy_label):
        noisy_percentage = int(noisy_label * self.n_samples)
        self.noisy_X = deepcopy(self.X)
        self.noisy_Y = deepcopy(self.Y)
        self.noisy_Y[:noisy_percentage] = (self.noisy_Y[:noisy_percentage] + 1) % 2

        perm_index = torch.randperm(self.noisy_X.size()[0])
        self.noisy_X = self.noisy_X[perm_index]
        self.noisy_Y = self.noisy_Y[perm_index]

    def __getitem__(self, index):
        return self.noisy_X[index], self.noisy_Y[index]

def batch_accuracy(y_pred_batch, y_batch):
    acc = ((y_pred_batch > 0) == y_batch).float().mean().item()
    return acc

def dataloader_accuracy(dataloader, model):
    model.eval()
    accuracy = []
    device = next(model.parameters()).device
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            y_pred = model(X_batch)[:, 0]
            batch_acc = batch_accuracy(y_pred, y_batch)
            accuracy.append(batch_acc)
    model.train()
    if len(accuracy) == 0:
        return 0
    return sum(accuracy) / len(accuracy)


