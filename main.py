from utils import jsonToDict
import numpy as np
import wandb
from train import LanguageModel

if __name__ == '__main__':
	seq_len = 6
	path = f'data/{seq_len}-seq_len/'

	## Getting vocabulary
	encoder = jsonToDict(path + 'encoder.json')
	decoder = jsonToDict(path + 'decoder.json')

	print(f'Vocab size: {len(encoder)}')

	data = np.load(path + 'data.npy')
	np.random.shuffle(data)
	split = 0.85
	idx = int(len(data) * split)
	train_set = data[:idx]
	test_set = data[idx:len(data)]

	print(len(train_set), len(test_set))

	params = {
		'seq_len':seq_len, ## Need this only to compare runs for wandb
		'num_probs':len(encoder),
		'emb_dim':128,
		'h_dim':512,
		'num_rnn':3,
		'dropout':0.2,
		'lr':0.001,
		'batch_size':512,
		}

	run = wandb.init(
		project=f'TextGenerator-{seq_len}_seq_len', 
		config=params, 
		# name=f'seq_len-{seq_len}-h_dim-{h_dim}-num_rnn-{num_rnn}-batch_size-{batch_size}',
		reinit=True)
	
	lm = LanguageModel(params)

	print(lm.decode(decoder, data[0]))

	lm.train(train_set, test_set, num_epochs=5)
	run.finish()

	# hdims = [64, 256, 512]
	# num_rnns = [2, 3, 4]
	# batch_sizes = [256, 512]

	# for h_dim in h_dims:
	# 	for num_rnn in num_rnns:
	# 		for batch_size in batch_sizes:
	# 			## Hyperparameters
	# 			params = {
	# 				'num_probs':len(encoder),
	# 				'emb_dim':128,
	# 				'h_dim':h_dim,
	# 				'num_rnn':num_rnn,
	# 				'dropout':0.2,
	# 				'lr':0.001,
	# 				'batch_size':batch_size,
	# 				}

	# 			run = wandb.init(
	# 				project='TextGenerator', 
	# 				config=params, 
	# 				name=f'seq_len-{seq_len}-h_dim-{h_dim}-num_rnn-{num_rnn}-batch_size-{batch_size}', reinit=True)
				
	# 			lm = LanguageModel(params)
	# 			lm.train(train_set, test_set, num_epochs=5)
	# 			run.finish()
