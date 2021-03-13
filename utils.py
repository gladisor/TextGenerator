import gutenbergpy.textget
import nltk
import json
import numpy as np
import os

class Book:
	def __init__(self, gutenberg_id):
		## Get, decode, and lowercase a book fromm gutenberg
		self.text = gutenbergpy.textget.strip_headers(
			gutenbergpy.textget.get_text_by_id(gutenberg_id)
			).decode('utf-8').lower()

		## Remove some unwanted characters
		self.remove_punctuation()

		## Create a list of chunks from the text
		self.chunks = nltk.blankline_tokenize(self.text)

	def remove_punctuation(self):
		## These pieces of punctuation generally just take up space
		## and dont provide much value
		self.text = self.text.replace('-', ' ')
		self.text = self.text.replace('—', ' ')
		self.text = self.text.replace('_', ' ')
		self.text = self.text.replace('*', ' ')

	def tokenize(self):
		self.chunks = list(map(lambda x: nltk.word_tokenize(x), self.chunks))

	def pad_chunks(self, seq_len):
		left_pad = ['<start_chunk>']
		right_pad = ['<end_chunk>']

		self.chunks = list(map(lambda chunk: left_pad * seq_len + chunk + right_pad, self.chunks))

	def build_vocab(self):
		self.vocab = set(sum(self.chunks, []))

def generate_examples(seq_len, chunk):
	examples = []
	for i in range(len(chunk) - seq_len):
		train = chunk[i : i + seq_len]
		label = chunk[i + seq_len]
		train.append(label)
		examples.append(train)
	return examples

def dictToJson(data, path):
	with open(path, 'w') as f:
		json.dump(data, f)

def jsonToDict(path):
	with open(path) as f:
		data = json.load(f)
	return data

def generate_dataset(seq_len):
	books = {
		'War and Peace':(2600, 386),
		'Anna Karenina':(1399, 6)
	}

	vocab = set()
	chunks = []

	## Creating location to store datset
	path = 'data/' + str(seq_len) + '-seq_len/'
	os.makedirs(path, exist_ok=True)

	## Extracting chunks from books
	for title, (ID, start) in books.items():
		print('Downloading: ' + title)
		book = Book(gutenberg_id=ID)
		del book.chunks[0:start]

		print('Tokenizing: ' + title)
		book.tokenize()

		print('Padding chunks: ' + title)
		book.pad_chunks(seq_len)

		print('Building vocabulary: ' + title)
		book.build_vocab()

		vocab = vocab.union(book.vocab)
		print('Total vocabulary size = ' + str(len(vocab)))

		chunks += book.chunks

	## Hashing chunks and generating training examples
	decoder = dict(enumerate(vocab))
	encoder = {v:k for k, v in decoder.items()}
	## Saving vocabulary
	dictToJson(decoder, path + 'decoder.json')
	dictToJson(encoder, path + 'encoder.json')

	hash_chunk = lambda chunk: [encoder[word] for word in chunk]

	print('Hashing dataset')
	chunks = list(map(hash_chunk, chunks))

	print('Generating training data')
	## Creating training examples of seq_len prediction words
	## and one target word
	data = list(
		map(
			generate_examples, 
			[seq_len] * len(chunks), 
			chunks
			)
		)

	## Combining all training exampes into one array
	## Target word is the last column
	data = sum(data, [])
	data = np.array(data)
	print(f'{len(data)} training examples')

	np.save(path + str(seq_len) + '-seq_len.npy', data)

if __name__ == '__main__':

	seq_len = 20
	generate_dataset(seq_len)