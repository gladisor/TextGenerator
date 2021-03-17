from utils import jsonToDict
from train import LanguageModel
import torch
import re

if __name__ == '__main__':
	import textstat
	import tqdm

	seq_len = 6
	path = f'data/{seq_len}-seq_len/'

	## Getting vocabulary
	encoder = jsonToDict(path + 'encoder.json')
	decoder = jsonToDict(path + 'decoder.json')

	## Getting params for model
	params = jsonToDict(path + 'model_params.json')

	## Creating language model object and loading trained model
	lm = LanguageModel(params)
	lm.model.load_state_dict(torch.load(path + 'model4.pt'))

	for _ in range(10):
		seq = lm.generate(encoder, decoder, seq_len=seq_len, num_words=50)
		seq = re.sub('\s+([.,:;?!])', '', seq)
		print(seq)
		print()

	# ## Generating and scoring sequences
	# score = []
	# for i in tqdm.tqdm(range(100)):
	# 	seq = lm.generate(encoder, decoder, seq_len=seq_len, num_words=50)
	# 	score.append(textstat.flesch_reading_ease(seq))

	# print(score)
	# print(sum(score)/len(score))