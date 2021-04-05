import os

import torch
import matplotlib.pyplot as plt
import torch.utils.data

from torch import nn


class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, y_pred, y):
        loss = torch.sqrt(self.mse(y_pred, y) + self.eps)
        return loss


def load_data(data_class, path_to_data, input_size, output_size, features, targets, bs):
    dataset = data_class(path_to_data, input_size, output_size, features, targets)

    train_len = int(len(dataset) * 0.9)
    val_len = len(dataset) - train_len

    train_set, val_set = torch.utils.data.random_split(dataset, [train_len, val_len])

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=bs, shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=bs, shuffle=True, num_workers=0)

    return train_loader, val_loader


def train(args, model, train_loader, val_loader, name="TCN"):
    print(name)
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Using GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    if not os.path.exists("models"):
        os.makedirs("models")
    if not os.path.exists("figures"):
        os.makedirs("figures")

    optimizer = torch.optim.Adam(list(model.parameters()), lr=args.lr,
                                 weight_decay=args.wd)

    model = model.to(device)
    loss = RMSELoss()

    train_loss = []
    val_loss = []
    best_loss = 0
    best_epoch = 0

    train_len = len(train_loader)
    val_len = len(val_loader)

    for epoch in range(args.epochs):
        model.train(True)
        epoch_train_loss = 0

        # Training
        for data in train_loader:
            x = data["input"].type(torch.FloatTensor).to(device)  # Load Input data
            label = data["label"].type(torch.FloatTensor).to(device)  # Load labels
            if args.output_size == 1:
                label = torch.squeeze(label, 1)
            optimizer.zero_grad()  # Reset gradients
            pred = model(x)  # Forward Pass
            batch_loss = loss(pred, label)  # Compute loss
            epoch_train_loss += batch_loss.item() / train_len

            batch_loss.backward()  # Backpropagation
            optimizer.step()  # Optimization

        train_loss.append(epoch_train_loss)

        # Validation
        model.train(False)
        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for data in val_loader:
                x = data["input"].type(torch.FloatTensor).to(device)  # Load Input data
                label = data["label"].type(torch.FloatTensor).to(device)  # Load labels
                if args.output_size == 1:
                    label = torch.squeeze(label, 1)
                optimizer.zero_grad()  # Reset gradients
                pred = model(x)  # Forward Pass
                batch_loss = loss(pred, label)  # Compute loss
                epoch_val_loss += batch_loss.item() / val_len
            val_loss.append(epoch_val_loss)

            if best_epoch == 0 or epoch_val_loss < best_loss:
                best_loss = epoch_val_loss
                best_epoch = epoch
                torch.save(model.state_dict(), "models/{}.pth".format(name))

        print("Epoch: {:3d} | Train loss: {:.3f} | Val loss: {:.3f}".format(epoch, train_loss[epoch],
                                                                                val_loss[epoch]))

    print("Training Complete")
    print("Best Validation Error ({}) at epoch {}".format(best_loss, best_epoch))

    # Plot Final Training Errors
    fig, ax = plt.subplots(figsize=(15, 9))
    plt.grid(True)
    ax.plot(train_loss, linewidth=2, label="Training Loss")
    ax.plot(val_loss, linewidth=2, label="Validation Loss")
    ax.set_title("{} Training & Validation Losses".format(name))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("RMSE Loss")
    ax.legend()
    fig.savefig("figures/{}_loss.png".format(name))
    fig.show()
    print("Finished Training!")
