import torch
import torch.nn as nn

class SeqToProb(nn.Module):
	"""
	Model which takes in a sequence of integers (tokens) and returns
	a discrete probability distribution over the same range of integers.
	"""
	def __init__(self, num_probs, emb_dim, h_dim, num_lstm, dropout, lr):
		super(SeqToProb, self).__init__()

		self.embedding = nn.Embedding(num_probs, emb_dim)

		self.lstm = nn.LSTM(
			input_size=emb_dim,
			hidden_size=h_dim,
			num_layers=num_lstm,
			batch_first=True,
			dropout=dropout
			)

		self.fc = nn.Linear(h_dim, num_probs)

		self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
		self.to(self.device)

		self.opt = torch.optim.Adam(self.parameters(), lr=lr)

	def forward(self, x):
		emb = self.embedding(x)
		out, (h_t, c_t) = self.lstm(emb)
		probs = self.fc(out[:, -1, :])
		return probs