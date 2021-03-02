import nltk
f = open('data/war_and_peace.txt', encoding='utf8')
text = f.read().lower()
# text = 'help-fix this, sentance-thing'
text = text.replace('-', ' ')
text = text.replace('—', ' ')

# print(text)
tokens = nltk.tokenize.word_tokenize(text)
unique = set(tokens)
print(unique)
print(len(unique))

# for line in f:
# 	print(line)

"""
Line numbering for pandas
Header: 0-17
Table of Contents: 18-400
"""

# import nltk
# f = open('data/war_and_peace.txt', encoding='utf8')
# text = f.read().lower()
# tokens = nltk.tokenize.word_tokenize(text)
# unique = set(tokens)
# print(len(unique))