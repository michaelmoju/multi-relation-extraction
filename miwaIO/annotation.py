import numpy as np


class EntityMention:
	def __init__(self, id, type, start, end, words):
		self.id = id
		self.type = type
		self.start = int(start)
		self.end = int(end)
		self.words = words


class RelationMention:
	def __init__(self, id, type, arg1, arg2):
		self.id = id
		self.type = type

		split = lambda arg: arg.split(":")
		self.arg1 = split(arg1)[1]
		self.arg2 = split(arg2)[1]


class EntityInstance:
	def __init__(self, sentence, mentions):
		self.sentence = sentence
		self.mentions = mentions

	def get_label(self, label2idx, max_sent_len):
		out_array = np.zeros((max_sent_len,))
		for i in range(len(self.sentence.tokens)):
			out_array[i] = label2idx['O']
		for m in self.mentions:
			for t_index, t in enumerate(self.sentence.tokens):
				if t.start == m.start:
					mention_token_start = t_index
				if t.end == m.end:
					mention_token_end = t_index
			assert mention_token_end >= mention_token_start

			for i in range(mention_token_start, mention_token_end+1):
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

		return out_array

	def get_tokens(self):
		return [t.word for t in self.sentence.tokens]


class RelationInstance:
	pass