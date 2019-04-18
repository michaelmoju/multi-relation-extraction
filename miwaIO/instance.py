import numpy as np
from word_embeddings import get_idx_sequence


def mention_to_token_span(m, sentence):
	for t_index, t in enumerate(sentence.tokens):
		if t.start == m.start:
			mention_token_start = t_index
		if t.end == m.end:
			mention_token_end = t_index
	assert mention_token_end >= mention_token_start
	return mention_token_start, mention_token_end

class EntityInstance:
	def __init__(self, sentence, mentions):
		self.sentence = sentence
		self.mentions = mentions

	def get_label(self, label2idx, max_sent_len):
		out_array = np.zeros((max_sent_len,))
		for i in range(len(self.sentence.tokens)):
			out_array[i] = label2idx['O']
		for m in self.mentions:
			mention_token_start, mention_token_end = mention_to_token_span(m, self.sentence)
			for i in range(mention_token_start, mention_token_end + 1):
				if mention_token_start == mention_token_end:
					key = 'U'
				elif i == mention_token_end:
					key = 'L'
				elif i == mention_token_start:
					key = 'B'
				else:
					key = 'I'
				key = key + '-' + m.type
				out_array[i] = label2idx[key]
		self.label = out_array

		return out_array

	def get_tokens(self):
		return [t.word for t in self.sentence.tokens]

	def get_word_idx(self, max_sent_len, word2idx):
		out_array = np.zeros((max_sent_len,))
		token_word_idx = get_idx_sequence(self.get_tokens(), word2idx)
		out_array[:len(token_word_idx)] = token_word_idx
		return out_array


class RelationInstance:
	def __init__(self, sentence, r_type, arg1, arg2):
		self.sentence = sentence
		self.type = r_type
		self.arg1 = arg1
		self.arg2 = arg2

	def get_label_position(self):
		arg1_array = np.ones(len(self.sentence.tokens),)
		arg2_array = np.ones(len(self.sentence.tokens),)

		arg1_token_start, arg1_token_end = mention_to_token_span(self.arg1, self.sentence)
		arg2_token_start, arg2_token_end = mention_to_token_span(self.arg2, self.sentence)

		arg1_array[arg1_token_start:arg1_token_end+1] = 2
		arg2_array[arg2_token_start:arg2_token_end+1] = 2

		return arg1_array, arg2_array

	def get_tokens(self):
		return [t.word for t in self.sentence.tokens]

	def get_word_idx(self, max_sent_len, word2idx):
		out_array = np.zeros((max_sent_len,))
		token_word_idx = get_idx_sequence(self.get_tokens(), word2idx)
		out_array[:len(token_word_idx)] = token_word_idx
		return out_array