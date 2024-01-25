import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import random
import cv2
from pathlib import Path
import torch
import torch.nn as nn
from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.reducers import ThresholdReducer
from pytorch_metric_learning.regularizers import LpRegularizer
from pytorch_metric_learning import losses
from sklearn.model_selection import KFold
import pickle

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Device: {device}')

class SiameseNetwork(nn.Module):

  def __init__(self, num_ft, width):
    super(SiameseNetwork, self).__init__()
    # Defining the fully connected layers
    self.fc1 = nn.Sequential(
     nn.LayerNorm(num_ft),
     # First Dense Layer
     nn.Linear(num_ft, width),
     nn.ReLU(),
     nn.BatchNorm1d(width),
     nn.Dropout(p=0.7),
     # Second Dense Layer
     nn.Linear(width, int(width/2)),
     nn.ReLU(),
     nn.BatchNorm1d(int(width/2)),
     nn.Linear(int(width/2), int(width/4)),
     nn.BatchNorm1d(int(width/4)),
     nn.ReLU(),
     nn.Linear(int(width/4), int(width/8)),
     nn.BatchNorm1d(int(width/8)),
     nn.ReLU(),
     nn.Linear(int(width/8), int(width/64)),
     nn.Sigmoid(),
     nn.BatchNorm1d(int(width/64))
     )

  def forward_once(self, x):
    # Forward pass 
    output = self.fc1(x.float())
    return output

  def forward(self, input1, input2, input3):
    # forward pass of input 1
    output1 = self.forward_once(input1)
    # forward pass of input 2
    output2 = self.forward_once(input2)
    # forward pass of input 3
    output3 = self.forward_once(input3)
    # returning the feature vectors of two inputs
    return [output1, output2, output3]

class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)


###############################################################################
# train and predict
def tnp(feature, embs_lst):
  num_ft = embs_lst.shape[1]
  ###############################################################################
  # create training set

  # pad training set with some negatives from test set
  test_padding = 0
  a_emb = np.zeros([train_len + test_padding, num_ft])
  p_emb = np.zeros([train_len + test_padding, num_ft])
  n_emb = np.zeros([train_len + test_padding, num_ft])

  for i in range(train_len):
      anchor = triplet_lst[i][0]
      positive = triplet_lst[i][1]
      negative = triplet_lst[i][2]

      a_emb[i,:] = embs_lst[anchor]
      p_emb[i,:] = embs_lst[positive]
      n_emb[i,:] = embs_lst[negative]

  a_emb = torch.from_numpy(a_emb)
  p_emb = torch.from_numpy(p_emb)
  n_emb = torch.from_numpy(n_emb)

  batch_size = 32
  dataset = ConcatDataset(a_emb, p_emb, n_emb)
  dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

  siamese_network = SiameseNetwork(num_ft, width)
  # define distance function, if we want to use it
  dist = torch.nn.PairwiseDistance(p=1.0, eps=1e-06, keepdim=True)
  dcos = torch.nn.CosineSimilarity(dim=1, eps=1e-08)
  # Loss function, optimizer and scheduler
  loss_func = nn.TripletMarginLoss(margin=2.0, p=2)
  optimizer = torch.optim.Adam(siamese_network.parameters(),
                              lr=lr,
                              weight_decay=wd)
  # load model / start main training loop
  save_path = f'./models/model-{feature}_lr-{lr}_wd-{wd}_width-{width}_epochs-{num_epochs}.pth'
  if load_model:
      print(f'Loading model: {save_path}')
      siamese_network.load_state_dict(torch.load(save_path))
  else:
      print('Beginning training now!')
      # siamese_network.load_state_dict(torch.load(save_path))
      for epoch in range(num_epochs):
        print(f'Epoch {epoch}')
        for i, (a, p, n) in enumerate(dataloader, 0):
          optimizer.zero_grad()
          a.to(device)
          p.to(device)
          n.to(device)
          a_out, p_out, n_out = siamese_network(a, p, n)
          # calculate loss, backpropagate
          loss = loss_func(a_out, p_out, n_out)
          loss.backward()
          optimizer.step()

          if i%100 == 0:
            print(f'Batch: {i}, Loss: {loss.item()}')
      torch.save(siamese_network.state_dict(), f'./models/model-{feature}_lr-{lr}_wd-{wd}_width-{width}_epochs-{num_epochs}.pth')


  ############################################################################
  # Repeat with test data
  a_tst = np.zeros([test_len, num_ft])
  p_tst = np.zeros([test_len, num_ft])
  n_tst = np.zeros([test_len, num_ft])

  for i in range(test_len):
      anchor = test_lst[i][0]
      positive = test_lst[i][1]
      negative = test_lst[i][2]

      a_tst[i,:] = embs_lst[anchor]
      p_tst[i,:] = embs_lst[positive]
      n_tst[i,:] = embs_lst[negative]


  a_tst = torch.from_numpy(a_tst)
  p_tst = torch.from_numpy(p_tst)
  n_tst = torch.from_numpy(n_tst)
  test_dataset = ConcatDataset(a_tst, p_tst, n_tst)
  test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

  with torch.no_grad():
    for i, (a, p, n) in enumerate(test_dataloader):
      a_out, p_out, n_out = siamese_network(a, p, n)

      dist1 = dist(a_out, p_out)
      dist0 = dist(a_out, n_out)
      
      dist_comp = torch.gt(dist0, dist1)

      dcos1 = dcos(a_out, p_out)
      dcos0 = dcos(a_out, n_out)

      cos_comp = torch.gt(dcos1, dcos0) # needs to be switched from what is normal

      vote1 = torch.abs(torch.sub(a_out, p_out))
      vote0 = torch.abs(torch.sub(a_out, n_out))

      element_vote = torch.gt(vote0, vote1)
      majority_vote = torch.mode(element_vote, 1)

      if i == 0:
        result_dcos = cos_comp
        result_dist = dist_comp
        result_vote = majority_vote.values
      else:
        result_dcos = torch.cat((result_dcos, cos_comp), dim=0)
        result_dist = torch.cat((result_dist, dist_comp), dim=0)
        result_vote = torch.cat((result_vote, majority_vote.values), dim=0)

  result_dcos = result_dcos.cpu().numpy()*1
  result_dist = result_dist.cpu().numpy()*1
  result_vote = np.transpose(result_vote.cpu().numpy())*1
  return result_vote, result_dist.reshape(test_len,), result_dcos.reshape(test_len,)

###############################################################################
# load data
# triplet list
train = 'train_triplets.txt'
triplet_lst = np.loadtxt(train, dtype=int)
train_len = round(len(triplet_lst))

test = 'test_triplets.txt'  
test_lst = np.loadtxt(test, dtype=int)
test_len = round(len(test_lst))


# features
inc = 'features/features_inception.pckl'
f = open(inc, 'rb')
inc_lst = pickle.load(f)
print(f'InceptionNet features: {inc_lst.shape}')
print(inc_lst)

res = pd.read_csv('features/image_features_resnet.csv')
res = res.sort_index(ascending=True, axis=1)
res = res.drop('.DS_S', axis=1)
res = res.T
res_lst = res.to_numpy()
print(f'ResNet features: {res_lst.shape}')
print(res_lst)

mbl = pd.read_csv('features/image_features_mobilenet.csv')
mbl = mbl.sort_index(ascending=True, axis=1)
mbl = mbl.drop('.DS_S', axis=1)
mbl = mbl.T
mbl_lst = mbl.to_numpy()
print(f'MobileNet features: {mbl_lst.shape}')
print(mbl_lst)

features = [inc_lst, res_lst, mbl_lst, res_lst, inc_lst, mbl_lst]
###############################################################################
# define hyperparameters
lr = 0.0001 # from grid search, this or 0.0005 were good
wd = 0 # from grid search, this made little difference
width = 2048 # from grid search, this or 1024, but this seems just a bit better
num_epochs = 6
load_model = False

###############################################################################
# generate predictions
dvote_sum = np.zeros(test_len)
dcomp_sum = np.zeros(test_len)
ccomp_sum = np.zeros(test_len)
for l, ftr in enumerate(features):
  if l>2:
    load_model = False
  dist_vote, dist_comp, cos_comp = tnp(l, ftr)
  dvote_sum = np.add(dvote_sum, dist_vote)
  dcomp_sum = np.add(dcomp_sum, dist_comp)
  ccomp_sum = np.add(ccomp_sum, cos_comp)

print(dvote_sum)  
print(dcomp_sum)
print(ccomp_sum)

dvote_res = [0 if dvote_sum[i] < 3 else 1 for i in range(test_len)]
dcomp_res = [0 if dcomp_sum[i] < 4 else 1 for i in range(test_len)]
ccomp_res = [0 if ccomp_sum[i] < 3 else 1 for i in range(test_len)]

total = np.add(dvote_sum, dcomp_sum)
total = np.add(total, ccomp_sum)
print(total)
total_res = [0 if total[i] < 9 else 1 for i in range(test_len)]

np.savetxt('predictions/many_dist.txt', dvote_res, fmt='%i')
np.savetxt('predictions/many_vote.txt', dcomp_res, fmt='%i')
np.savetxt('predictions/many_cos.txt', ccomp_res, fmt='%i')

np.savetxt('predictions/many_total.txt', total_res, fmt='%i')
