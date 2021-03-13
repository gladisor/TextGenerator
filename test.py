import numpy as np
from utils import jsonToDict
import torch
import torch.nn as nn
from models import SeqToProb

## Loading in sequence data
data = np.load('data/10-seq_len/10-seq_len.npy')
megabytes = (data.size * data.itemsize)/1e6
print(f'Data is {megabytes} megabytes')

## Splitting data into X and y columns
data = torch.LongTensor(data)
X = data[:, :-1]
y = data[:, -1]

## Getting vocabulary
encoder = jsonToDict('data/10-seq_len/encoder.json')
decoder = jsonToDict('data/10-seq_len/decoder.json')

num_probs = len(encoder)
emb_dim = 50
h_dim = 20
num_lstm = 1
seq_len = 10
lr = 0.0001

loss = nn.CrossEntropyLoss()

## Creating model
model = SeqToProb(
	num_probs=num_probs,
	emb_dim=emb_dim,
	h_dim=h_dim,
	num_lstm=num_lstm,
	seq_len=seq_len,
	lr=lr)

train_X = X[0:5].to(model.device)
train_y = y[0:5].to(model.device)

y_hat = model(train_X)

print(y_hat)
print(loss(y_hat, train_y))