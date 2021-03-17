import json
import gutenbergpy.textget
import nltk
from collections import Counter
import os
from pathlib import Path
import random
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

def write_sequences(list_of_strings, path):
	with open(path, 'w', encoding='utf8') as f:
		for line in list_of_strings:
			f.write(' '.join(line) + '\n\n')

def read_sequences(path):
	## Reading file to string
	with open(path, 'r', encoding='utf8') as f:
		sequences = f.read()

	## Tokenizing string based on blankline
	sequences = nltk.blankline_tokenize(sequences)

	## Tokenizing each item in sequence into words & punctuation
	sequences = list(map(nltk.word_tokenize, sequences))
	return sequences

class Book:
	def __init__(self, gutenberg_id):
		## Get, decode, and lowercase a book fromm gutenberg
		self.text = gutenbergpy.textget.strip_headers(
			gutenbergpy.textget.get_text_by_id(gutenberg_id)
			).decode('utf-8').lower()

	def clean(self, min_seq=4):
		## Remove unwanted punctuation
		## ['“', '”', '"', '"', '(', ')'] <-- keep for later
		remove_chars = ['-', '—', '_', '*', '“', '”', '"', '"']
		for char in remove_chars:
		    self.text = self.text.replace(char, ' ')

		## Seperate book into list of paragraphs
		self.sequences = nltk.blankline_tokenize(self.text)

def extract_raw_data(ids, data_folder, sequences_folder, min_seq_len=4, validation_split=0.05):
	## List to hold all sequences
	sequences = []

	## Extracting sequences from books
	for ID in ids:
		book = Book(gutenberg_id=ID)

		## Cleaning / tokenizing
		book.clean()
		sequences += book.sequences

	## Creating vocabulary
	sequences = list(map(nltk.word_tokenize, sequences))
	sequences = list(filter(lambda x: len(x) > min_seq_len, sequences))

	vocab = Counter()
	for seq in sequences:
		vocab.update(seq)

	## Saving reversable vocabulary
	decoder = dict(enumerate(vocab.keys()))
	encoder = {v:k for k, v in decoder.items()}
	dictToJson(decoder, data_folder / 'decoder.json')
	dictToJson(encoder, data_folder / 'encoder.json')

	## Reserving sequences for validation
	random.shuffle(sequences)
	split = int(len(sequences) * validation_split)
	valid = sequences[0:split]
	train = sequences[split:len(sequences)]

	## Creating location to store datset
	path = data_folder / sequences_folder
	os.makedirs(path, exist_ok=True)

	## Writing training sequences
	write_sequences(train, path / 'train_sequences.txt')

	## Writing testing sequences with a blank line in between
	write_sequences(valid, path / 'valid_sequences.txt')

def create_dataset(num_predict_words, name):
	## These ids correspond to 4 of Tolstoys novels on gutenberg.org
	ids = [2600, 1399, 243, 6157]
	data_folder = Path('data/').resolve()
	sequences_folder = 'sequences/'
	path = data_folder / sequences_folder

	## If the main dataset doesnt exist: build it
	if not os.path.exists(path):
		extract_raw_data(ids, data_folder, sequences_folder)

	## Getting train sequences
	train = read_sequences(path / 'train_sequences.txt')

	## Getting valid sequences
	valid = read_sequences(path / 'valid_sequences.txt')

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

if __name__ == '__main__':
	create_dataset(
		num_predict_words=4,
		name='4_predict_words')
