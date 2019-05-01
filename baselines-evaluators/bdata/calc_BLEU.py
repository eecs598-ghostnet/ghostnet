from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.bleu_score import sentence_bleu
import sys

corpus_filename = 'lyrics.txt'

def txt_to_list(filename):
	corpus = open(filename, 'r')
	lines = corpus.readlines()

	ref_list = []
	ref = []
	for i in range(len(lines)-1):
		if (len(lines[i]) < 3 and len(lines[i+1]) < 3):
			ref_list.append(ref)
			ref.clear()
		else:
			words = lines[i].split()
			for word in words:
				ref.append(word)

	ref_list.append(ref)

	corpus.close()

	return ref_list

def txt_to_sentence(filename):
	corpus = open(filename, 'r')
	lines = corpus.readlines()

	ref_list = []
	for line in lines:
		if (len(line) > 2):
			ref_list.append(line.split())

	corpus.close()

	return ref_list

def get_bleu(filename):
	corpus_list = txt_to_sentence(corpus_filename)
	file_list = txt_to_sentence(filename)
	print("Corpus length: " + str(len(corpus_list)))
	print("File Length: " + str(len(file_list)))
	#references = [corpus_list]
	references = corpus_list

	score = 0.0
	filelen = 0.0
	find_sim = 0.0
	not_sim = 0.0

	for filetext in file_list:
		#canidate = [filetext]
		canidate = filetext
		#score += corpus_bleu(references, canidate)
		sentence_score = sentence_bleu(references, canidate)
		score += sentence_score
		if (sentence_score > 0.5):
			find_sim += 1.0
		else:
			not_sim += 1.0
		filelen += 1.0
		if (filelen % 25 == 0.0):
			print("Done with " + str(filelen) + " files.")
			print("Score: " + str(score/filelen))
			print("Sim Percent: " + str(find_sim/filelen))
			print("Not Sim Percent: " + str(not_sim/filelen))
		if (filelen == 125):
			break;


	score = score/filelen

	return score


if (len(sys.argv) == 1):
	print("Please enter file name!")
	exit(0)
filename = sys.argv[1]
print(get_bleu(filename))


