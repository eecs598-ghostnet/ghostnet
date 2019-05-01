import numpy as np
import os
import sys
from scipy import spatial

word_freq_file = 'corp_word_freq.txt'
doc_freq_file = 'corp_doc_freq.txt'

def calc_cos_sim(filename):
	# Load corpus tf and idf data
	word_freq = open(word_freq_file, 'r')
	doc_freq = open(word_freq_file, 'r')

	word_freq_lines = word_freq.readlines()
	doc_freq_lines = doc_freq.readlines()

	if (len(word_freq_lines) != len(doc_freq_lines)):
		print("Not an equal number of tf and idf's for corpus, exiting")
		exit(0)

	key_words = []
	tf_corpus = []
	idf_corpus = []

	for i in range(len(word_freq_lines)):
		tf_data = word_freq_lines[i].split()
		idf_data = doc_freq_lines[i].split()
		key = tf_data[0]
		tf_word = tf_data[1]
		idf_word = idf_data[1]
		key_words.append(key)
		tf_corpus.append(float(tf_word))
		idf_corpus.append(float(idf_word))

	n_corpus = sum(tf_corpus) # total number of words in the corpus
	adtf_corpus = np.array(tf_corpus, dtype=float) / float(n_corpus)

	word_freq.close()
	doc_freq.close()

	print("Done loading corpus data")

	# Load file to compare to data
	file = open(filename, 'r')

	file_lines = file.readlines()
	word_occur = [0] * len(tf_corpus)
	n_file_words = 0

	for line in file_lines:
		words = line.split()
		for word in words:
			if word in key_words:
				idx = key_words.index(word)
				word_occur[idx] += 1
				n_file_words += 1

	file.close()

	print("Done loading file data")

	# Get cosine similarity
	adword_occur = np.array(word_occur, dtype=float) / float(n_file_words)

	tfidf_corpus = np.multiply(adtf_corpus, idf_corpus)
	tfidf_file = np.multiply(adword_occur, idf_corpus)

	cos_sim = 1 - spatial.distance.cosine(tfidf_corpus, tfidf_file)

	return cos_sim

if (len(sys.argv) == 1):
	print("Please enter file name!")
	exit(0)
filename = sys.argv[1]
print(calc_cos_sim(filename))

