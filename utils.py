import gutenbergpy.textget
import nltk

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
		examples.append([train, label])
	return examples

def generate_dataset(seq_len):
	books = {
		# 'War and Peace':(2600, 386),
		'Anna Karenina':(1399, 6)
	}

	vocab = set()
	chunks = []

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
	vocab = dict(enumerate(vocab))
	vocab = {v:k for k, v in vocab.items()}

	hash_chunk = lambda chunk: [vocab[word] for word in chunk]

	print('Hashing dataset')
	chunks = list(map(hash_chunk, chunks))

	print('Generating training data')
	data = list(
		map(
			generate_examples, 
			[seq_len] * len(chunks), 
			chunks
			)
		)

if __name__ == '__main__':
	import numpy as np

	seq_len = 5
	# generate_dataset(seq_len)

	data = [
		[748, 748, 748, 748, 748, 10051, 8630, 9909, 6237, 8630, 4141, 412, 2253, 4452, 7379, 7477, 4942, 12120, 9523, 3398, 12005, 6237, 10051, 2157, 14, 4225, 4410, 12805, 12389, 5685, 412, 12185, 2982, 2157, 12185, 10292, 3725, 7477, 722], 
		[748, 748, 748, 748, 748, 4942, 7783, 12389, 8630, 4141, 6237, 8830, 6933, 6237, 2157, 8728, 12389, 2253, 6816, 412, 3802, 4220, 12389, 453, 2270, 3725, 1532, 3329, 1033, 6237, 1532, 3886, 1557, 6237, 1532, 11325, 5786, 8449, 4190, 1310, 3725, 722]
		]

	train = list(
		map(
			generate_examples,
			[seq_len] * len(data),
			data
			)
		)

	train = sum(train, [])
	train = np.array(train)
	print(train)