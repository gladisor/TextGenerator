import json
import gutenbergpy.textget
import nltk
from collections import Counter
import os
from pathlib import Path
import random
import numpy as np

import torch

def dictToJson(dictionary, path):
	## Stores a python dict to a .json file
	with open(path, 'w') as f:
		json.dump(dictionary, f)

def jsonToDict(path):
	## Returns a python dict from a .json file
	with open(path) as f:
		dictionary = json.load(f)
	return dictionary

def text_to_sequences(text):
	## Tokenizing string based on blankline (paragraphs)
	sequences = nltk.blankline_tokenize(text)

	## If we want sentance level data uncomment:
	# sequences = list(map(nltk.sent_tokenize, sequences))
	# sequences = sum(sequences, [])

	## Tokenizing each item in sequence into words & punctuation
	sequences = list(map(nltk.wordpunct_tokenize, sequences))
	return sequences

def write_sequences(list_of_strings, path):
	with open(path, 'w', encoding='utf8') as f:
		for line in list_of_strings:
			f.write(' '.join(line) + '\n\n')

def read_sequences(path):
	## Reading file to string
	with open(path, 'r', encoding='utf8') as f:
		text = f.read()

	sequences = text_to_sequences(text)
	return sequences

def pad_sequences(num_predict_words, sequences):
	## Apply padding to each sequence on both ends

	start = ['<sos>'] * num_predict_words
	end = ['<eos>']
	pad = lambda x: start + x + end

	sequences = list(map(pad, sequences))
	return sequences

def encode_sequences(encoder, sequences):
	## Turns sequences containing words into sequences of integers
	return list(map(lambda seq: [encoder[word] for word in seq], sequences))

def decode_sequences(decoder, sequences):
	return list(map(lambda seq: [decoder[str(num)] for num in seq], sequences))

def chunk_sequence(num_predict_words, seq):
	examples = []
	for i in range(len(seq) - num_predict_words):
		examples.append(seq[i : i + num_predict_words + 1])
	return examples

def generate_examples(num_predict_words, sequences):
	data = []
	for seq in sequences:
		data += chunk_sequence(num_predict_words, seq)
	return data

class Book:
	def __init__(self, gutenberg_id):
		## Get, decode, and lowercase a book fromm gutenberg
		self.text = gutenbergpy.textget.strip_headers(
			gutenbergpy.textget.get_text_by_id(gutenberg_id)
			).decode('utf-8').lower()

	def clean(self, min_seq_len=4):
		## Remove unwanted punctuation
		## ['“', '”', '"', '"', '(', ')'] <-- keep for later
		remove_chars = ['-', '—', '_', '*', '“', '”', '"', '"', '‘', '’', '\'']
		for char in remove_chars:
		    self.text = self.text.replace(char, ' ')

		## Generate sequences from text
		self.sequences = text_to_sequences(self.text)

		## Remove sequences < min_seq_len
		self.sequences = list(filter(lambda x: len(x) > min_seq_len, self.sequences))


def extract_raw_data(ids):
	## List to hold all sequences
	sequences = []

	## Extracting sequences from books
	for ID in ids:
		book = Book(gutenberg_id=ID)

		## Cleaning / tokenizing
		book.clean()
		sequences += book.sequences

	return sequences

def build_vocabulary(sequences, special_tokens=['<sos>', '<eos>']):
	## Creating vocabulary
	vocab = Counter()
	for seq in sequences:
		vocab.update(seq)

	## Adding start and end of sequence tokens
	vocab.update(special_tokens)

	## Saving reversable vocabulary
	decoder = dict(enumerate(vocab.keys()))
	encoder = {v:k for k, v in decoder.items()}

	return (decoder, encoder)

def save_data(path, sequences, vocab, split={'test':0.10, 'valid':0.05}):
	## Creating location to store datset
	os.makedirs(path, exist_ok=True)

	## Reserving sequences for validation
	random.shuffle(sequences)
	valid_split = int(len(sequences) * split['valid'])
	test_split = int(len(sequences) * split['test'])
	mid = valid_split + test_split

	valid = sequences[0 : valid_split]
	test = sequences[valid_split : mid]
	train = sequences[mid : len(sequences)]

	## Validation
	write_sequences(valid, path / 'valid.txt')
	## Testing
	write_sequences(test, path / 'test.txt')
	## Training
	write_sequences(train, path / 'train.txt')

	## Saving vocab
	decoder, encoder = vocab
	dictToJson(decoder, path / 'decoder.json')
	dictToJson(encoder, path / 'encoder.json')

def sequences_pipeline(path, num_predict_words, encoder):
	data = read_sequences(path)
	data = pad_sequences(num_predict_words, data)
	data = encode_sequences(encoder, data)
	data = generate_examples(num_predict_words, data)
	random.shuffle(data)
	return np.array(data)

def create_dataset(num_predict_words):
	## Main file to house all data derived from the data in main/
	data_path = Path('data/').resolve()
	## File to keep the unprocessed data in
	main_path = 'main/'
	path = data_path / main_path

	## If the main dataset doesnt exist: build it
	if not os.path.exists(path):
		## These ids correspond to 4 of Tolstoys novels on gutenberg.org
		ids = [2600, 1399, 243, 6157]
		sequences = extract_raw_data(ids)
		## vocab is a tuple: (decoder, encoder)
		vocab = build_vocabulary(sequences)
		save_data(path, sequences, vocab)

	## Creating location to store this processed data
	os.makedirs(data_path / str(num_predict_words), exist_ok=False)

	encoder = jsonToDict(path / 'encoder.json')
	file_names = ['train', 'test', 'valid']

	## Passing data through pipeline
	for file_name in file_names:
		data = sequences_pipeline(path / (file_name + '.txt'), num_predict_words, encoder)
		np.save(data_path / str(num_predict_words) / (file_name + '.npy'), data)

class TextData(torch.utils.data.Dataset):
	def __init__(self, data):
		super(TextData, self).__init__()
		## Casting incoming data as a LongTensor
		data = torch.LongTensor(data)

		## Splitting data into columns
		self.prediction_words = data[:, :-1]
		self.target_word = data[:, -1]

	def __len__(self):
		return len(self.target_word)

	def __getitem__(self, idx):
		return self.prediction_words[idx], self.target_word[idx]
