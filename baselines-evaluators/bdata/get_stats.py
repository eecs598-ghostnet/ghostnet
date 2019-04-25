from matplotlib import pyplot as plt
import os
import numpy as np

artist_file_ns = 'nslyrics.txt'
artist_file = 'lyrics.txt'
word_freq_file = 'word_freq.txt'
doc_freq_file = 'doc_freq.txt'

# Retruns the number of occurence of each word in the lyrics
def plotWordFrequency(input):
    f = open(artist_file_ns,'r')
    o = open(word_freq_file,'w')
    words = [x for y in [l.split() for l in f.readlines()] for x in y]
    alldata = sorted([(w, words.count(w)) for w in set(words)], key = lambda x:x[1], reverse=True)
    data = alldata[:40] 
    most_words = [x[0] for x in data]
    times_used = [int(x[1]) for x in data]
    for point in alldata:
    	outputstr = str(point[0]) + ' ' + str(point[1]) + '\n'
    	o.write(outputstr)
    plt.figure(figsize=(20,10))
    plt.bar(x=sorted(most_words), height=times_used, color = 'grey', edgecolor = 'black',  width=.5)
    plt.xticks(rotation=45, fontsize=18)
    plt.yticks(rotation=0, fontsize=18)
    plt.xlabel('Most Common Words:', fontsize=18)
    plt.ylabel('Number of Occurences:', fontsize=18)
    plt.title('Most Commonly Used Words: %s' % (artist_file), fontsize=24)
    plt.show()
    f.close()
    o.close()

def getIDF():
	# fetch words
	word_freq = open(word_freq_file,'r')
	words = [] #get all the words from the frequency file!
	lines = word_freq.readlines()

	for line in lines:
		data = line.split()
		word = data[0]
		words.append(word)

	word_freq.close()
	print("Loaded words!")

	# get IDF
	f = open(artist_file,'r')
	lines = f.readlines()
	idf = [0] * len(words)
	wordOccur = [False] * len(words)
	numDocuments = 0

	for i in range(len(lines) - 1):
		# New document!
		if (len(lines[i]) < 3 and len(lines[i+1]) < 3):
			for j in range(len(wordOccur)):
				if wordOccur[j]:
					idf[j] += 1
			wordOccur = [False] * len(words)
			numDocuments += 1
			if numDocuments % 10 == 0:
				print("Wrote document " + str(numDocuments))
		else:
			line_words = lines[i].split()
			for j in range(len(words)):
				if words[j] in line_words:
					wordOccur[j] = True

	f.close()

	# write IDF
	print("Documents: " + str(numDocuments))
	o = open(doc_freq_file,'w')
	for i in range(len(words)):
		if idf[i] == 0:
			cur_idf = 1
		else:
			cur_idf = idf[i]
		true_idf = 1 + np.log(numDocuments/cur_idf)
		outputstr = words[i] + ' ' + str(true_idf) + '\n'
		o.write(outputstr)

	o.close()

getIDF()

