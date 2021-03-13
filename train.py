from utils import jsonToDict
from models import SeqToProb
from data import TextData

import torch.nn as nn
from torch.utils.data import DataLoader

import tqdm

if __name__ == '__main__':
	## Getting vocabulary
	encoder = jsonToDict('data/20-seq_len/encoder.json')

	## Hyperparameters
	num_probs = len(encoder)
	emb_dim = 100
	h_dim = 100
	num_lstm = 1
	dropout = 0.0
	lr = 0.001
	batch_size = 512
	num_epochs = 10

	## Choosing loss
	CE = nn.CrossEntropyLoss()

	## Creating model
	model = SeqToProb(
		num_probs=num_probs,
		emb_dim=emb_dim,
		h_dim=h_dim,
		num_lstm=num_lstm,
		dropout=dropout,
		lr=lr)

	## Creating dataset
	data = TextData('data/20-seq_len/20-seq_len.npy')
	loader = DataLoader(data, num_workers=1, shuffle=True)

	for epoch in range(num_epochs):
		total_loss = 0
		num_batches = 0

		for X, y in tqdm.tqdm(loader):
			X, y = X.cuda(), y.cuda()
			
			model.opt.zero_grad()

			y_hat = model(X)
			loss = CE(y_hat, y)
			
			loss.backward()
			model.opt.step()

			total_loss += loss.item()
			num_batches += 1

		print(f'Average loss: {total_loss / num_batches}')