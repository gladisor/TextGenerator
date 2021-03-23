from pathlib import Path
import numpy as np
import wandb
import torch
torch.manual_seed(0)

from train import LanguageModel
from data import jsonToDict

if __name__ == '__main__':
	data_path = Path('data/').resolve()
	main_path = data_path / 'main/'

	## Getting vocabulary
	encoder = jsonToDict(main_path / 'encoder.json')
	decoder = jsonToDict(main_path / 'decoder.json')

	## Getting common parameters for each model
	params = {
		'seq_len':num_predict_words, ## Need this only to compare runs for wandb
		'num_probs':len(encoder),
		'emb_dim':128,
		'h_dim':512,
		'num_rnn':3,
		'dropout':0.2,
		'lr':0.001,
		'batch_size':512,
		}

	for num_predict_words in range(2, 9):
		# num_predict_words = 7
		model_path = data_path / str(num_predict_words)

		print(f'Vocab size: {len(encoder)}')

		train = np.load(model_path / 'train.npy')
		test = np.load(model_path / 'test.npy')

		print(len(train), len(test))

		run = wandb.init(
			project=f'TextGenerator-{num_predict_words}_seq_len',
			config=params,
			name=str(num_predict_words) + 'redo',
			reinit=True)

		lm = LanguageModel(params)

		print(lm.decode(decoder, test[0]))

		lm.train(train, test, num_epochs=5, path=model_path)
		run.finish()

	## Hyperparameter sweep
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
