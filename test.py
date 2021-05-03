from pathlib import Path
from data import \
		jsonToDict, sequences_pipeline, \
		TextData, read_sequences, \
		pad_sequences, encode_sequences, \
		chunk_sequence, write_sequences

from train import LanguageModel
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd

if __name__ == '__main__':
	data_path = Path('data/').resolve()
	main_path = data_path / 'main/'

	## Getting vocabulary
	encoder = jsonToDict(main_path / 'encoder.json')
	decoder = jsonToDict(main_path / 'decoder.json')

	## Geting validation data
	# valid = read_sequences(main_path / 'valid.txt')

	## Loss for perplexity calculation
	CE = nn.CrossEntropyLoss()

	# results = pd.DataFrame({'num_predict_words':[], 'perplexity':[]})

	for num_predict_words in [6]: #[2, 3, 4, 5, 6, 7, 8]:
		## Preparing validation data
		# padded = pad_sequences(num_predict_words, valid)
		# encoded = encode_sequences(encoder, padded)

		## Getting params for model
		model_path = data_path / str(num_predict_words)
		params = jsonToDict(model_path / 'model_params.json')

		## Creating language model object and loading trained model
		lm = LanguageModel(params)
		lm.model.load_state_dict(torch.load(model_path / 'model4.pt'))

		# perplexity = 0
		# for seq in encoded:
		# 	data = chunk_sequence(num_predict_words, seq)
		# 	data = torch.LongTensor(data)
		# 	data = TextData(data)
		#
		# 	x, y = data[0:len(data)]
		# 	x, y = x.to(lm.model.device), y.to(lm.model.device)
		#
		# 	with torch.no_grad():
		# 		logits = lm.model(x)
		# 		loss = CE(logits, y)
		# 		perplexity += torch.exp(loss).item()
		#
		# perplexity = perplexity / len(encoded) ## <-- averageing perplexity over number of sequences
		#
		# results = results.append({'num_predict_words':num_predict_words, 'perplexity':perplexity}, ignore_index=True)

		text = ['well', ',', 'prince', ',', 'so', 'genoa']
		text = ['and', 'twisting', 'the', 'ramrod', 'he', 'looked']
		enc = torch.LongTensor([lm.encode(encoder, text)])
		out = lm.model(enc.cuda())
		out = torch.softmax(out, dim=-1)
		word = torch.argmax(out, dim=-1)

		words = ['at', 'ahead', 'gloomily']
		for word in words:
			idx = encoder[word]
			print(out[0, idx])

		# print(decoder[str(word.item())])

		# total_num_words = 0
		# generated_text = []
		#
		# while total_num_words < 1500:
		# 	text = lm.generate(encoder, decoder, num_predict_words, num_words=600)
		# 	total_num_words += len(text)
		# 	generated_text.append(text)
		#
		# print(num_predict_words, total_num_words)
		#
		# write_sequences(generated_text, model_path / 'generated.txt')

	# print(results)
	# results.to_csv('results.csv')
