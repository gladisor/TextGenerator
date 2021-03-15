from utils import jsonToDict, dictToJson
from models import SeqToProb
from data import TextData
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import wandb
import tqdm

class LanguageModel:
	def __init__(self, params):
		self.params = params

		self.model = SeqToProb(
			num_probs=params['num_probs'],
			emb_dim=params['emb_dim'],
			h_dim=params['h_dim'],
			num_rnn=params['num_rnn'],
			dropout=params['dropout'],
			lr=params['lr'])

		self.criterion = nn.CrossEntropyLoss()

	def decode(self, decoder, sequence):
		return [decoder[str(word)] for word in sequence]

	def encode(self, encoder, sequence):
		return [encoder[word] for word in sequence]

	def accuracy(self, probs, y):
		predicted_words = torch.argmax(probs, dim=-1)

		total_correct = (predicted_words == y).sum()
		return total_correct/predicted_words.shape[0]

	def perplexity(self, probs, y):
		pass

	def generate(self, encoder, decoder, seq_len, num_words):
		stop_words = ['.', '!', ':']
		seq = ['<s>'] * seq_len

		for i in range(num_words):
			x = self.encode(encoder, seq[i:i+seq_len])
			x = torch.LongTensor([x]).to(self.model.device)

			logits = self.model(x)
			probs = torch.softmax(logits, dim=-1)

			dist = torch.distributions.Categorical(probs)
			word = dist.sample().item()

			word = decoder[str(word)]

			seq.append(word)

			print(word, end=' ')
			
			if word in stop_words:
				break

	def train(self, train, test, num_epochs):

		## Putting data into loaders
		train_loader = DataLoader(
			TextData(train), 
			batch_size=self.params['batch_size'], 
			num_workers=1, 
			shuffle=True)

		test_loader = DataLoader(
			TextData(test), 
			batch_size=self.params['batch_size'], 
			num_workers=1, 
			shuffle=True)

		## Begin training
		for epoch in range(num_epochs):
			## Train cycle
			self.model.train()
			train_loss = 0
			train_batches = 0

			for X, y in tqdm.tqdm(train_loader):
				X, y = X.cuda(), y.cuda()
				self.model.opt.zero_grad()

				## Forward
				logits = self.model(X)
				loss = self.criterion(logits, y)

				## Backward
				loss.backward()
				self.model.opt.step()

				## Measure stats
				train_loss += loss.item()
				train_batches += 1

				## Log train loss for each batch so we get a good graph
				wandb.log({'train_loss':loss.item()})

			## Testing cycle
			self.model.eval()
			test_loss = 0
			test_acc = 0
			test_batches = 0

			for X, y in tqdm.tqdm(test_loader):
				X, y = X.cuda(), y.cuda()

				## Forward
				logits = self.model(X)
				loss = self.criterion(logits, y)

				## Measure stats
				test_loss += loss.item()
				probs = torch.softmax(logits, dim=-1)
				test_acc += self.accuracy(probs, y)
				test_batches += 1

			## Report statistics
			avg_train_loss = train_loss / test_batches
			avg_test_loss = test_loss / test_batches
			avg_test_acc = test_acc / test_batches

			print(f'\
				Train loss: {train_loss / train_batches}, Test loss: {avg_test_loss}, Test acc: {avg_test_acc}')

			wandb.log({
				'test_loss':avg_test_loss,
				'test_acc':avg_test_acc})

			## Save model each epoch
			torch.save(self.model.state_dict(), f'model{epoch}.pt')

		## Save model parameters at the end
		dictToJson(self.params, 'model_params.json')

if __name__ == '__main__':
	seq_len = 4 
	path = f'data/{seq_len}-seq_len/'

	## Getting vocabulary
	encoder = jsonToDict(path + 'encoder.json')
	decoder = jsonToDict(path + 'decoder.json')

	params = jsonToDict(path + 'model_params.json')

	lm = LanguageModel(params)
	lm.model.load_state_dict(torch.load(path + 'model4.pt'))

	lm.generate(encoder, decoder, seq_len=seq_len, num_words=50)