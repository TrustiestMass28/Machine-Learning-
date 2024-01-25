import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import KFold

# autoencoder class
class LP(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder1 = nn.Linear(
            in_features=kwargs["input_shape"], out_features=128
        )
        self.encoder2 = nn.Linear(
            in_features=128, out_features=128
        )
        self.output_layer = nn.Linear(
            in_features=128, out_features=1
        )

    def forward(self, features):
        activation = self.encoder1(features)
        activation = torch.relu(activation)
        code = self.encoder2(activation)
        code = torch.relu(code)
        output = self.output_layer(code)
        return output

# create model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Running on {device}.')
model = LP(input_shape=1000).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# import pretrain data: 50,000 data points, 1000 features 
pre_data = pd.read_csv('./csv/pretrain_features.csv', index_col=0)
pre_data = pre_data.drop('smiles', axis=1).to_numpy()
pre_lbls = pd.read_csv('./csv/pretrain_labels.csv', index_col=0)
pre_lbls = pre_lbls.to_numpy()

# make a dataloader
data_tensor = torch.Tensor(pre_data)
lbls_tensor = torch.Tensor(pre_lbls)
pre_dataset = TensorDataset(data_tensor, lbls_tensor)
pre_dataloader = DataLoader(pre_dataset, batch_size=32)

train_model = True
epochs = 25
model_path = f'./models/lumo_predictor_epochs-{epochs}.pth'
if train_model:
    print('training model.')
    for epoch in range(epochs):
        loss = 0
        for batch_features, batch_labels in pre_dataloader:
            # load features
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            
            # zero optimizer
            optimizer.zero_grad()

            # calculate outputs
            outputs = model(batch_features)

            # calculate loss
            train_loss = criterion(outputs, batch_labels)

            # backpropogate
            train_loss.backward()

            # backpropogate
            optimizer.step()

            loss += train_loss.item()

        loss = loss / len(pre_dataloader)

        print(f'epoch: {epoch}, loss: {loss}')
    torch.save(model.state_dict(), model_path)
    print('model successfully saved.')
else:
    print('loading model.')
    model.load_state_dict(torch.load(model_path))
    with torch.no_grad():
        for batch_features, batch_labels in pre_dataloader:
            # load features
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            # calculate outputs
            outputs = model(batch_features)

print(f'output tensor: {outputs.cpu().detach().numpy()}')
print(f'labels: {pre_lbls}')
