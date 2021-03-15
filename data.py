import numpy as np
import torch

class TextData(torch.utils.data.Dataset):
	"""docstring for TextData"""
	def __init__(self, data):
		super(TextData, self).__init__()
		## Splitting data into columns
		data = torch.LongTensor(data)
		# idx = torch.randperm(data.shape[0])
		# data = data[idx].view(data.size())

		self.prediction_words = data[:, :-1]
		self.target_word = data[:, -1]

	def __len__(self):
		return len(self.target_word)

	def __getitem__(self, idx):
		return self.prediction_words[idx], self.target_word[idx]

