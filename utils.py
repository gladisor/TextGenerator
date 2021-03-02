import requests

def download_book(url, filename):
	r = requests.get(url)
	open('data/' + filename, 'wb').write(r.content)

if __name__ == '__main__':

	## War and Peace
	url = 'http://www.gutenberg.org/files/2600/2600-0.txt'

	download_book(url, 'war_and_peace.txt')
	
	## Anna Karenina
	url = 'http://www.gutenberg.org/files/1399/1399-0.txt'

	download_book(url, 'anna_karenina.txt')