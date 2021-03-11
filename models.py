import torch
import torch.nn as nn

class SeqToProb(nn.Module):
	"""
	Model which takes in a sequence of integers (tokens) and returns
	a discrete probability distribution over the same range of integers.
	"""
	def __init__(self, num_probs, emb_dim, h_dim, num_lstm, seq_len):
		super(SeqToProb, self).__init__()

		self.embedding = nn.Embedding(num_probs, emb_dim)

		self.lstm = nn.LSTM(
			input_size=emb_dim,
			hidden_size=h_dim,
			num_layers=num_lstm,
			batch_first=True)

		self.fc = nn.Linear(seq_len * h_dim, num_probs)


	def forward(self, x):
		emb = self.embedding(x)
		out, _ = self.lstm(emb)
		out = out.reshape(out.shape[0], out.shape[1] * out.shape[2])
		out = torch.softmax(self.fc(out), dim=-1)
		return out

if __name__ == '__main__':

	model = SeqToProb(
		num_probs=10,
		emb_dim=5,
		h_dim=2,
		num_lstm=1,
		seq_len=3)

	x = torch.LongTensor([
		[1, 2, 7],
		[3, 4, 8],
		[9, 6, 5],
		[4, 9, 2]
		])

	out = model(x)
	print(x.shape, out.shape)
	print(out)