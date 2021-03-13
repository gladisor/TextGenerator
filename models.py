import torch
import torch.nn as nn

class SeqToProb(nn.Module):
	"""
	Model which takes in a sequence of integers (tokens) and returns
	a discrete probability distribution over the same range of integers.
	"""
	def __init__(self, num_probs, emb_dim, h_dim, num_lstm, seq_len, lr):
		super(SeqToProb, self).__init__()

		self.embedding = nn.Embedding(num_probs, emb_dim)

		self.lstm = nn.LSTM(
			input_size=emb_dim,
			hidden_size=h_dim,
			num_layers=num_lstm,
			batch_first=True)

		self.fc = nn.Linear(seq_len * h_dim, num_probs)

		self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
		self.to(self.device)

		self.opt = torch.optim.Adam(self.parameters(), lr=lr)

	def forward(self, x):
		emb = self.embedding(x)
		out, _ = self.lstm(emb)
		out = out.reshape(out.shape[0], out.shape[1] * out.shape[2])
		out = torch.softmax(self.fc(out), dim=-1)
		return out