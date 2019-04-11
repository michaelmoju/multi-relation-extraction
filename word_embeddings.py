# Author: Moju Wu

import numpy as np
import re

all_zeros = "ALL_ZERO"
unknown = "UNKNOWN"

special_tokens = {"-DQS-": "``", "-DQE-": "''", "-LRB-": "(", "-RRB-": ")", "-COMMA-": ",", "-COLON-": ":",
				  "-SEMICOLON-": ";", "-PERIOD-": ".", "-DOLLAR-": "$", "-PERCENT-": "%"}


def get_idx(word, word2idx):

	unknown_idx = word2idx[unknown]
	word = word.strip()

	if word in word2idx:
		return word2idx[word]
	elif word.lower() in word2idx:
		return word2idx[word.lower()]
	elif word.upper() in word2idx:
		return word2idx[word.upper()]
	elif word in special_tokens:
		return word2idx[special_tokens[word]]

	trimmed = re.sub("(^\W|\W$)", "", word)
	if trimmed in word2idx:
		return word2idx[trimmed]
	elif trimmed.lower() in word2idx:
		return word2idx[trimmed.lower()]
	no_digits = re.sub("([0-9][0-9.,]*)", '0', word)
	if no_digits in word2idx:
		return word2idx[no_digits]
	return unknown_idx


def get_idx_sequence(word_sequence, word2idx):
	"""
	    Get embedding indices for the given word sequence.

	    :param word_sequence: sequence of words to process
	    :param word2idx: dictionary of word mapped to their embedding indices
	    :return: a sequence of embedding indices
	    """
	vector = []
	for word in word_sequence:
		word_idx = get_idx(word, word2idx)
		vector.append(word_idx)
	return vector


def load_word_emb(emb_path):
	word2idx = {}
	embeddings = []

	with open(emb_path, 'r', encoding='utf-8') as f:
		idx = 1
		for l in f:
			split = l.strip().split(' ')
			word2idx[split[0]] = idx
			embeddings.append(np.array([float(num) for num in split[1:]]))
			idx += 1

	word_emb_size = embeddings[0].shape[0]

	word2idx[all_zeros] = 0
	embeddings = np.asarray([[0.0] * word_emb_size] + embeddings, dtype='float32')

	unknown_emb = np.average(embeddings[idx - 101:idx - 1, :], axis=0)
	embeddings = np.append(embeddings, [unknown_emb], axis=0)

	word2idx[unknown] = idx

	print('word embedding size:' + str(embeddings.shape))
	print('word2idx size: ' + str(len(word2idx)))

	return embeddings, word2idx


if __name__ == '__main__':
	load_word_emb('../resource/embeddings/glove/glove.6B.50d.txt')
