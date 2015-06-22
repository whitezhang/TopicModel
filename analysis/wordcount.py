import sys
import os
import re
import numpy as np
from operator import itemgetter
import operator
import stemmer

STOP_WORDS_DIC = set()
def load_stop_words(sw_file_path):
	sw_file = open(sw_file_path, "r")
	for word in sw_file:
		word = word.replace("\n", "")
		word = word.replace("\r\n", "")
		STOP_WORDS_DIC.add(word)
	sw_file.close()

# di
class Document(object):
	PUNCTUATION = ['(', ')', ':', ';', ',', '-', '!', '.', '?', '/', '"', '*', '>', '<']
	CARRIAGE_RETURNS = ['\n', '\r\n']
	WORD_REGEX = "^[a-z']+$"

	def __init__(self, filepath):
		self.filepath = filepath
		self.file = open(self.filepath)
		self.lines = []
		self.words = []

	def split(self, STOP_WORDS_DIC):
		self.lines = [line for line in self.file]
		for line in self.lines:
			words = line.split(' ')
			for word in words:
				clean_word = self._clean_word(word)
				if clean_word and (clean_word not in STOP_WORDS_DIC) and (len(clean_word) > 1):
					self.words.append(clean_word)
		self.file.close()

	def _clean_word(self, word):
		word = word.lower()
		for punc in Document.PUNCTUATION + Document.CARRIAGE_RETURNS:
			word = word.replace(punc, '').strip("'")
			# stemmer: dogs -> dog ; created -> creat
			ps = stemmer.PorterStemmer()
			word = ps.stem(word, 0, len(word)-1)
		return word if re.match(Document.WORD_REGEX, word) else None

class Corpus(object):
	def __init__(self):
		self.documents = []
		self.words_frequency = {}

	def add_document(self, document):
		self.documents.append(document)

	def build_vocabulary(self):
		for document in self.documents:
			for word in document.words:
				if word in self.words_frequency:
					self.words_frequency[word] += 1
				else:
					self.words_frequency[word] = 1
		# self.vocabulary = frequent_words_set

def main(argv):
	data_path = '../plsa/data/'
	load_stop_words(data_path+'stopwords.txt')

	corpus = Corpus()
	document_path = data_path+'20_newsgroups/'

	for root, dirs, files in os.walk(document_path):
		for name in files:
			document_file = root + '/' + name
#			print document_file
			document = Document(document_file)
			document.split(STOP_WORDS_DIC)
			corpus.add_document(document)
	corpus.build_vocabulary()
	# Compute Zipf values
	y_value = {}
	for k, v in corpus.words_frequency.items():
		if v in y_value:
			y_value[v] = y_value[v]+1
		else:
			y_value[v] = 1
	sorted_y_value = sorted(y_value.items(), key=operator.itemgetter(1))
	for p in sorted_y_value:
		print p[0], p[1]


if __name__ == '__main__':
	main(sys.argv)