import numpy as np
import sys
import random
from scipy import spatial

song_file_name = 'temp.txt'
corpus_idf_file_name = 'corp_doc_freq.txt'
corpus_file_name = '../../data/lyrics/combined/lyrics.txt'
sim_tf_file_name = 'corp_sim_freq.txt'
sim_tfidf_file_name = 'corp_tfidf.txt'
sample_prop = 1.00
cap_verse = 100

def dict_to_text(input_dict, outfile):
	for entry in input_dict:
		outputstr = entry + ' ' + str(input_dict[entry]) + '\n'
		outfile.write(outputstr)
	outfile.write('\n')

def text_to_dict(text_input):
	outDict = {}
	entries = text_input.split('\n')
	for entry in entries:
		if len(entry) > 1:
			data = entry.split()
			outDict[data[0]] = float(data[1])

	return outDict

def versedict_to_alldict(versDict, allDict):
	outDict = {}
	for outEntry in allDict:
		if outEntry in versDict:
			outDict[outEntry] = versDict[outEntry]
		else:
			outDict[outEntry] = 0.0

	return outDict

def get_verseTF(corpus_file_n, sim_out_file_n):
	corpus_in = open(corpus_file_n, 'r')
	tf_out = open(sim_out_file_n, 'w')
	corpus = corpus_in.read()
	verses = corpus.split('\n\n\n')
	verses_done = 0
	for verse in verses:
		verses_done += 1
		tot_words = 0.0
		verse_tf = {}
		verse_words = verse.split()
		for word in verse_words:
			if len(word) < 1:
				continue
			if word in verse_tf:
				verse_tf[word] += 1
			else:
				verse_tf[word] = 1

			tot_words += 1.0
		for word in verse_tf:
			verse_tf[word] = float(verse_tf[word]) / tot_words

		dict_to_text(verse_tf, tf_out)
		if verses_done % 2500 == 0:
			print(str('Done with ' + str(verses_done) + ' verses.'))

	corpus_in.close()
	tf_out.close()

def load_song_tf(song_file_n, allDict):
	song_tf = []
	song_txt_file = open(song_file_n, 'r')
	song_txt = song_txt_file.read()

	songs = song_txt.split('\n\n')
	for song in songs:
		tot_words = 0.0
		words = song.split()
		song_tf_dict = {}
		for word in words:
			if word in song_tf_dict:
				song_tf_dict[word] += 1.0
			else:
				song_tf_dict[word] = 1.0
			tot_words += 1.0
		for word in song_tf_dict:
			song_tf_dict[word] = float(song_tf_dict[word]) / float(tot_words)
		tf_dict = versedict_to_alldict(song_tf_dict, allDict)
		song_tf.append(tf_dict)

	return song_tf

	song_txt_file.close()

def calc_cossim(vec1, vec2):
	return 1 - spatial.distance.cosine(vec1, vec2)

def dict_mult(dict1, dict2):
	outVec = [0.0]*len(dict1)
	if len(dict1) != len(dict2):
		print("Dict 1 and Dict 2 multiply not the same, exiting")
		exit(0)
	idx = 0
	for entry in dict1:
		outVec[idx] = dict1[entry] * dict2[entry]
		idx += 1

	return outVec

def calc_sim(song_file_n, sim_tf_file_n, corpus_idf_file_n, sim_tfidf_file_n):

	tf_in = open(sim_tf_file_n, 'r')
	all_tf = tf_in.read()

	# Load cord idf
	corp_in = open(corpus_idf_file_n, 'r')
	corp_idf_txt = corp_in.read()
	corp_idf = text_to_dict(corp_idf_txt)

	# Load song tf idf information
	songs_tf = load_song_tf(song_file_n, corp_idf)
	songs_tfidf = []
	sim_max = [0.0] * len(songs_tf)
	for song_tf in songs_tf:
		song_tfidf = dict_mult(song_tf, corp_idf)
		songs_tfidf.append(song_tfidf)

	# Run through each verse
	num_verses = 0
	verses_tf_txt = all_tf.split('\n\n')
	for verse_tf_txt in verses_tf_txt:
		if (random.random() > sample_prop):
			continue
		if (num_verses == cap_verse):
			break
		verse_tf_vdict = text_to_dict(verse_tf_txt)
		verse_tf = versedict_to_alldict(verse_tf_vdict, corp_idf)
		verse_tfidf = dict_mult(verse_tf, corp_idf)
		for song_id in range(len(songs_tfidf)):
			song_tfidf = songs_tfidf[song_id]
			cossim = calc_cossim(song_tfidf, verse_tfidf)
			if (cossim > sim_max[song_id]):
				sim_max[song_id] = cossim
		num_verses += 1

	for idx in range(len(sim_max)):
		print(sim_max[idx])

	tf_in.close()
	corp_in.close()


if (len(sys.argv) > 1):
	corpus_file_name = sys.argv[1]
	song_file_name = sys.argv[1]

# USE ONLY ONE!

# Load the song term frequencies
# get_verseTF(corpus_file_name, sim_tf_file_name)

# Get the similarity
calc_sim(song_file_name, sim_tf_file_name, corpus_idf_file_name, sim_tfidf_file_name)


