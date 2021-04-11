import os
import torch
import torch.nn as nn
import torch.utils.data
import math
import matplotlib.pyplot as plt
import numpy as np

from train_utils import RMSELoss, load_data
from attrdict import AttrDict
from config import *
from datasets import CaseUpc, Category, CaseUpcTV, CategoryTV


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        # pe.requires_grad = False
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class Transformer(nn.Module):
    def __init__(self, args, num_layers=3, dropout=0.1):
        super(Transformer, self).__init__()
        self.feature_size = args.num_features
        self.target_size = args.num_targets
        self.input_size = args.input_size
        self.output_size = args.output_size
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(self.feature_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.feature_size, nhead=4, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(self.feature_size, self.target_size)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        src = torch.transpose(src, 0, 1)
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        return torch.transpose(output, 0, 1)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


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
            label = torch.squeeze(label, 1)
            optimizer.zero_grad()  # Reset gradients
            pred = model(x)  # Forward Pass
            pred = pred[:, -args.output_size:]
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
                pred = pred[:, -args.output_size:]
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


def run(path_to_data):
    features = FEATURES
    targets = TARGETS

    # CaseUpc
    args = {
        'input_size': 50,  # decreasing didn't help
        'output_size': 12,
        'num_features': len(features),
        'num_targets': len(targets),
        'lr': 0.0001,
        'wd': 0.0005,
        'bs': 1000,
        'epochs': 200
    }

    args = AttrDict(args)
    name = "Transformer_upc"
    train_loader, val_loader = load_data(CaseUpcTV, path_to_data, args.input_size, args.output_size, features, targets,
                                         args.bs)
    model = Transformer(args)
    train(args, model, train_loader, val_loader, name)

    # Category
    args = {
        'input_size': 50,  # decreasing didn't help
        'output_size': 12,
        'num_features': len(features),
        'num_targets': len(targets),
        'lr': 0.0001,
        'wd': 0.0005,
        'bs': 32,
        'epochs': 200
    }

    args = AttrDict(args)
    name = "Transformer_category"

    train_loader, val_loader = load_data(CategoryTV, path_to_data, args.input_size, args.output_size, features,
                                         targets, args.bs)
    model = Transformer(args)
    train(args, model, train_loader, val_loader, name)


def run_single(path_to_data):
    features = FEATURES
    targets = TARGETS

    # CaseUpc
    args = {
        'input_size': 20,  # decreasing didn't help
        'output_size': 12,
        'num_features': len(features),
        'num_targets': len(targets),
        'lr': 0.00001,
        'wd': 0.0005,
        'bs': 16,
        'epochs': 200
    }

    args = AttrDict(args)
    dataset = CaseUpc(path_to_data, args.input_size, args.output_size, features, targets)

    top_cases = np.load("data/top_cases.npy")
    for case in top_cases:
        data = dataset.upc_to_ts[case]

        train_loader, val_loader = load_data(data, path_to_data, args.input_size, args.output_size, features,
                                             targets, args.bs, single=True)
        model = Transformer(args)
        name = "Transformer_upc_{}".format(case)
        train(args, model, train_loader, val_loader, name)

    # Category
    args = {
        'input_size': 20,
        'output_size': 12,
        'num_features': len(features),
        'num_targets': len(targets),
        'lr': 0.00001,
        'wd': 0.0005,
        'bs': 16,
        'epochs': 500
    }

    args = AttrDict(args)
    dataset = Category(path_to_data, args.input_size, args.output_size, features, targets)

    top_categories = np.load("data/top_categories.npy")

    for cat in top_categories:
        data = dataset.category_to_ts[cat]

        train_loader, val_loader = load_data(data, path_to_data, args.input_size, args.output_size, features,
                                             targets, args.bs, single=True)
        model = Transformer(args)
        name = "Transformer_category_{}".format(cat)
        train(args, model, train_loader, val_loader, name)
