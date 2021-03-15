import gutenbergpy.textget
import nltk
import json
import numpy as np
import os
import csv

class Book:
	def __init__(self, gutenberg_id):
		## Get, decode, and lowercase a book fromm gutenberg
		self.text = gutenbergpy.textget.strip_headers(
			gutenbergpy.textget.get_text_by_id(gutenberg_id)
			).decode('utf-8')

	def clean(self, seq_len, min_seq=4):
		print('--- Cleaning ---')
		## Remove unwanted punctuation
		remove_chars = ['-', '—', '_', '*', '“', '”', '"', '"', '(', ')']
		for char in remove_chars:
		    self.text = self.text.replace(char, ' ')

		## Seperate book into list of paragraphs
		paragraphs = nltk.blankline_tokenize(self.text)

		## Sepearte each paragraph into a list of sentances
		sentances = list(map(nltk.sent_tokenize, paragraphs))

		## Combine sentances into big list of all sentances
		sentances = sum(sentances, [])

		## Convert to lowercase
		sentances = list(map(str.lower, sentances))

		## Split sentances up into sequences of words
		sequences = list(map(nltk.word_tokenize, sentances))
		self.sequences = list(filter(lambda x: len(x) > min_seq, sequences))

		# Padding sequences to correct length
		pad = ['<s>']
		self.sequences = list(map(lambda x: pad * seq_len + x, self.sequences))

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
		'War and Peace':2600,
		'Anna Karenina':1399,
		'Various Shorts':243,
		'What Men Live By':6157
	}

	## List to hold all sequences
	sequences = []

	## Creating location to store datset
	path = 'data/' + str(seq_len) + '-seq_len/'
	os.makedirs(path, exist_ok=True)

	## Extracting sequences from books
	for title, ID in books.items():
		book = Book(gutenberg_id=ID)

		## Cleaning / tokenizing
		book.clean(seq_len)

		sequences += book.sequences

	## Saving unprocessed sequences
	print('--- Saving sequences ---')
	## Converting tokenized sequence into one seperated by spaces
	rows = list(map(lambda x: ' '.join(x), sequences))
	with open(path + 'sequences.txt', 'w') as f:
		for row in rows:
			f.write('%s\n' % row)

	## Creating vocab
	print('--- Creating vocabulary ---')
	vocab = set(sum(sequences, []))

	## Hashing chunks and generating training examples
	decoder = dict(enumerate(vocab))
	encoder = {v:k for k, v in decoder.items()}

	## Saving vocabulary
	dictToJson(decoder, path + 'decoder.json')
	dictToJson(encoder, path + 'encoder.json')

	## Converting sequences into training data
	print('--- Generating training data ---')
	hash_sequence = lambda x: [encoder[word] for word in x]
	sequences = list(map(hash_sequence, sequences))

	## Creating training examples of seq_len prediction word and one target word
	data = list(
		map(
			generate_examples, 
			[seq_len] * len(sequences), 
			sequences
			)
		)

	## Combining all training exampes into one array, target word is the last column
	data = sum(data, [])
	data = np.array(data)
	print(f'{len(data)} training examples')

	np.save(path + 'data.npy', data)

if __name__ == '__main__':
	seq_len = 4
	generate_dataset(seq_len)