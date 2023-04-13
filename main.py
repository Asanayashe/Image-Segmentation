import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from model import UNet
from dataset import Carvana


def show_image(img, mask, pred):
    fig, axes = plt.subplots(1, 3, figsize=(10, 5))
    axes[0].imshow(img.permute(1, 2, 0))
    axes[0].set_title('input image')
    axes[1].imshow(mask.permute(1, 2, 0))
    axes[1].set_title('original mask')
    axes[2].imshow(pred.permute(1, 2, 0))
    axes[2].set_title('predicted mask')
    fig.show()


def train(model, optimizer, criterion):
    train_losses = []
    val_lossess = []

    # calculate train epochs
    epochs = 10

    for epoch in range(epochs):
        model.train()
        train_total_loss = 0
        train_iterations = 0

        for img, mask in tqdm(train_dataloader):
            img = img.to(device)
            mask = mask.to(device)

            optimizer.zero_grad()
            # speed up the training
            pred = model(img)
            train_loss = criterion(pred, mask)
            train_total_loss += train_loss.item()

            train_loss.backward()
            optimizer.step()

        train_losses.append(train_total_loss)

        # evaluate mode
        model.eval()
        with torch.no_grad():
            val_total_loss = 0
            val_iterations = 0
            scores = 0

            for img, mask in tqdm(valid_dataloader):
                img = img.to(device)
                mask = mask.to(device)

                pred = model(img)
                val_iterations += 1
                val_loss = criterion(pred, mask)
                val_total_loss += val_loss.item()

            val_lossess.append(val_total_loss)

            # plot_train_progress(model)

            show_image(img[0].detach().cpu(), mask[0].detach().cpu(), pred[0].detach().cpu())

    return train_losses, val_lossess


if __name__ == "__main__":
    transforms = T.Compose([T.Resize((256, 256)), T.ToTensor()])
    train_dataset = Carvana('images/', transforms)
    valid_dataset = Carvana('images/', transforms, False)

    img, mask = train_dataset[0]
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img.permute(1, 2, 0))
    axes[1].imshow(mask.permute(1, 2, 0))
    fig.show()

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=8, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(in_channels=3, out_channels=1).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=10e-4)

    history = train(model, optimizer, criterion)
