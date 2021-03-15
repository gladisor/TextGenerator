import torch
import torch.nn as nn

class SeqToProb(nn.Module):
	"""
	Model which takes in a sequence of integers (tokens) and returns
	a discrete probability distribution over the same range of integers.
	"""
	def __init__(self, num_probs, emb_dim, h_dim, num_rnn, dropout, lr):
		super(SeqToProb, self).__init__()

		## Embedding integer as vectors
		self.embedding = nn.Embedding(num_probs, emb_dim)

		## Sequence processing layers
		self.lstm = nn.LSTM(
			input_size=emb_dim,
			hidden_size=h_dim,
			num_layers=num_rnn,
			batch_first=True,
			dropout=dropout)

		## Fully connected layer from hidden dim to number of probs
		self.fc1 = nn.Linear(h_dim, h_dim)
		self.drop1 = nn.Dropout(dropout)

		self.fc2 = nn.Linear(h_dim, num_probs)

		## Discovering device
		self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
		self.to(self.device)

		## Creating optimizer
		self.opt = torch.optim.Adam(self.parameters(), lr=lr)

	def forward(self, x):
		x = self.embedding(x)

		x, _ = self.lstm(x)

		x = torch.relu(self.fc1(x[:, -1, :]))
		x = self.drop1(x)

		logits = self.fc2(x)
		return logits

if __name__ == '__main__':

	num_probs = 10
	emb_dim = 2
	h_dim = 5
	num_lstm = 6
	dropout = 0.1
	lr = 0.001

	model = SeqToProb(
		num_probs, 
		emb_dim, 
		h_dim, 
		num_lstm, 
		dropout, 
		lr)

	x = torch.LongTensor([
		[1, 5, 9],
		[9, 3, 1],
		[5, 3, 0],
		[2, 5, 9]]).cuda()

	logits = model(x)