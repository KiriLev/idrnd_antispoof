import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from model import Model
from dataset import AntispoofDataset
from validation import validation


def save_model(model_, save_path, name_postfix=''):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    model_path = os.path.join(save_path, f"model_{name_postfix}.pt")

    torch.save(
        {
            'model': model_.state_dict(),
            },
        model_path
    )


def train():
    checkpoints_path = './checkpoints'
    num_epochs = 30
    batch_size = 50

    model = Model()
    model.train()
    model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.BCEWithLogitsLoss()

    train_paths = ['p']  # TODO replace
    val_paths = ['p']

    train_transform = None
    train_dataset = AntispoofDataset(paths=train_paths, transform=train_transform)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=50,
                              shuffle=True,
                              num_workers=4,
                              drop_last=True)

    val_dataset = AntispoofDataset(paths=val_paths, transform=train_transform)
    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=4,
                            drop_last=False)

    tq = None
    try:
        for epoch in range(num_epochs):

            print('Epoch {}/{}'.format(epoch, num_epochs - 1))

            tq = tqdm(total=len(train_loader) * batch_size)

            losses = []

            # iterate over data
            for inputs, labels in train_loader:
                inputs = inputs.cuda()
                labels = labels.cuda()

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                with torch.set_grad_enabled(True):
                    outputs = model(inputs).view(-1)
                    loss = criterion(outputs, labels.float())

                    # backward + optimize only if in training phase
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                    tq.update(batch_size)
                    losses.append(loss.item())

                # statistics

            epoch_loss = np.mean(losses)
            epoch_metrics = validation(model, val_loader=val_loader)

            print('Loss: {:.4f}\t Metrics: {}'.format(epoch_loss, epoch_metrics))
            save_model(model, checkpoints_path, name_postfix=f'e{epoch}')

    except KeyboardInterrupt:
        tq.close()
        print('Ctrl+C, saving model...')
        save_model(model, checkpoints_path)


if __name__ == '__main__':
    train()