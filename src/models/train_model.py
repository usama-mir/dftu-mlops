from torch.utils.data import DataLoader
from torch import nn
import torch
from model import Model
import click
from src.data.dataset import MyDataset
import matplotlib.pyplot as plt

@click.group()
def cli():
    pass

@click.command()
@click.option("--lr", default=1e-3, help='learning rate to use for training')
@click.option("--epochs", default=10, help="Number of epochs to run through - default is 10")
@click.option("--batch_size", default=10, help="Batchsize for training")
@click.option("--image_path", help="Path to training images")
@click.option("--label_path", help="Path to training labels")
def train(lr, epochs, image_path, label_path, batch_size):

    model = Model()
    train_loader = DataLoader(MyDataset(image_path, label_path), batch_size=batch_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    epoch = epochs
    train_loss = []

    for epoch in range(epoch):

        for data in train_loader:
            optimizer.zero_grad()
            x, y = data
            x = x.to(torch.float32)
            preds = model(x)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
        train_loss.append(loss.item())
        print(f"Epoch {epoch+1}/{epochs}. Loss: {loss}")

    torch.save(model.state_dict(), 'models/trained_model.pt')

    plt.plot(train_loss, '-')
    plt.xlabel('Training step')
    plt.ylabel('Training loss')
    plt.savefig("reports/figures/training_curve.png")

    return model

cli.add_command(train)

if __name__ == "__main__":
    cli()