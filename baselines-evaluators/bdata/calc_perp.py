import numpy as np
import sys

model_file_name = 'word_freq.txt'
input_file = 'nslyrics.txt'

def perplexity(input_text, model):
	perp = 0
	N = 0
	input_words = input_text.split()
	for word in input_words:
		if (len(word) > 0):
		    if word in model:
		    	N += 1
		    	perp = perp - np.log(model[word])

	perp = perp/N
	perp = np.exp(perp)
	return perp

def perplexity2(input_text, model):
	perp = 0.0
	N = 0
	input_words = input_text.split()
	word_dict = {}
	file_words = 0
	for word in input_words:
		if (len(word) > 0):
			file_words += 1
			if word in word_dict:
				word_dict[word] += 1
			else:
				word_dict[word] = 1

	for word in input_words:
		if word in model:
			perp = perp - (model[word])*(np.log(word_dict[word]/file_words))

	#perp = perp/N
	perp = np.exp(perp)
	return perp

def load_unigram(model_file):
	model = {}
	file_model = open(model_file, 'r')
	model_lines = file_model.readlines()
	total_freq = 0
	for line in model_lines:
		data = line.split()
		model[data[0]] = int(data[1])
		total_freq += int(data[1])

	for key in model:
		model[key] = model[key]/total_freq

	file_model.close()

	return model

def load_input(input_file, num_lines):
	input_file = open(input_file, 'r')
	input_lines = input_file.readlines()
	lines_read = 0
	outputstr = ''
	for line in input_lines:
		lines_read += 1
		outputstr += line 
		if lines_read == num_lines:
			break

	return outputstr

if (len(sys.argv) > 1):
	input_file = sys.argv[1]

model = load_unigram(model_file_name)
input_text = load_input(input_file, 100)
perp = perplexity(input_text, model)
perp2 = perplexity2(input_text, model)
print(perp)