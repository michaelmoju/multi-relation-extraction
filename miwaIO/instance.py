import numpy as np
from word_embeddings import get_idx_sequence
import my_models

e_type2idx = {'X':0, 'O': 1, 'PER': 2, 'ORG': 3, 'LOC': 4, 'GPE': 5, 'FAC': 6, 'VEH': 7, 'WEA': 8}
e_idx2type = {v: k for k, v in e_type2idx.items()}

e_label2idx = {'B-PER': 1, 'I-PER': 2, 'L-PER': 3, 'U-PER': 4,
               'B-ORG': 5, 'I-ORG': 6, 'L-ORG': 7, 'U-ORG': 8,
               'B-LOC': 9, 'I-LOC': 10, 'L-LOC': 11, 'U-LOC': 12,
               'B-GPE': 13, 'I-GPE': 14, 'L-GPE': 15, 'U-GPE': 16,
               'B-FAC': 17, 'I-FAC': 18, 'L-FAC': 19, 'U-FAC': 20,
               'B-VEH': 21, 'I-VEH': 22, 'L-VEH': 23, 'U-VEH': 24,
               'B-WEA': 25, 'I-WEA': 26, 'L-WEA': 27, 'U-WEA': 28, 'O': 0}
e_idx2label = {v: k for k, v in e_label2idx.items()}

r_label2idx = {'PHYS-lr': 1, 'PART-WHOLE-lr': 2, 'PER-SOC-lr': 3, 'ORG-AFF-lr': 4, 'ART-lr': 5, 'GEN-AFF-lr': 6,
               'PHYS-rl': 7, 'PART-WHOLE-rl': 8, 'PER-SOC-rl': 9, 'ORG-AFF-rl': 10, 'ART-rl': 11, 'GEN-AFF-rl': 12,
               'NONE': 0}
r_idx2label = {v: k for k, v in r_label2idx.items()}

p = my_models.p

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

	def get_label(self, max_sent_len):
		out_array = np.zeros((max_sent_len,))
		for i in range(len(self.sentence.tokens)):
			out_array[i] = e_label2idx['O']
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
				out_array[i] = e_label2idx[key]
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
	
	def get_one_hot_position(self, p, train_entity):
		if train_entity:
			arg1_array = np.zeros((p['max_sent_len'], p['lstm2']*2))
			arg2_array = np.zeros((p['max_sent_len'], p['lstm2']*2))
		else:
			arg1_array = np.zeros((p['max_sent_len'], p['word_emb']))
			arg2_array = np.zeros((p['max_sent_len'], p['word_emb']))
		
		arg1_token_start, arg1_token_end = mention_to_token_span(self.arg1, self.sentence)
		arg2_token_start, arg2_token_end = mention_to_token_span(self.arg2, self.sentence)
		
		arg1_array[arg1_token_start:arg1_token_end + 1, :] = 1
		arg2_array[arg2_token_start:arg2_token_end + 1, :] = 1
		
		return arg1_array, arg2_array

	def get_tokens(self):
		return [t.word for t in self.sentence.tokens]

	def get_word_idx(self, max_sent_len, word2idx):
		out_array = np.zeros((max_sent_len,))
		token_word_idx = get_idx_sequence(self.get_tokens(), word2idx)
		out_array[:len(token_word_idx)] = token_word_idx
		return out_array
	
	
class SentenceRelationInstance:
	"""
	Sentence Relation Instance: A sentence as a instance which includes all the relations and entities.
	"""
	
	def __init__(self, sentence, entity_lst, relation_lst, entity_dict):
		self.sentence = sentence
		self.entity_lst = entity_lst
		self.relation_mn_lst = relation_lst
		self.relation_ext_lst = []
		
		e_tuple_check_lst = []
		
		# append the RelationExtInstance with annotated relations
		for r in self.relation_mn_lst:
			arg1 = entity_dict[r.arg1]
			arg2 = entity_dict[r.arg2]
			
			if arg1.start < arg2.start:
				type = r.type + '-lr'
				e1 = arg1
				e2 = arg2
			elif arg1.start > arg2.start:
				type = r.type + '-rl'
				e1 = arg2
				e2 = arg1
			else:
				raise NotImplementedError
			
			self.relation_ext_lst.append(RelationExtInstance(type, e1, e2, self.sentence))
			e_tuple_check_lst.append((e1, e2))
		
		# append the RelationExtInstance with NONE annotated relations (NONE type)
		for arg1_idx in range(len(self.entity_lst)-1):
			for arg2_idx in range(arg1_idx+1, len(self.entity_lst)):
				e_tuple = (self.entity_lst[arg1_idx], self.entity_lst[arg2_idx])
				if e_tuple not in e_tuple_check_lst:
					
					# **Ignore the same-word entity pairs
					if e_tuple[0].words != e_tuple[1].words:
						self.relation_ext_lst.append(RelationExtInstance('NONE', e_tuple[0], e_tuple[1], self.sentence))
						e_tuple_check_lst.append(e_tuple)

	def dump(self):
		print('Sentence:')
		print(self.sentence)
		
		print('Entity List:')
		for e in self.entity_lst:
			print('\t' + str(e))
		
		print('Relation Extraction Instance List:')
		for index, r in enumerate(self.relation_ext_lst):
			print('relation{}'.format(index))
			print('\t' + str(r))
		print('==================')
		
			
class RelationExtInstance:
	"""
	Relation Extraction Instance: Include the NONE type and right to left or left to right relation type
	"""
	def __init__(self, type, e1, e2, sentence):
		self.type = type
		
		assert e1.start <= e2.start
		self.e1 = e1
		self.e2 = e2
		self.sentence = sentence
		
	def __str__(self):
		return 'e1:{}\t'.format(self.e1) + 'e2:{}\t'.format(self.e2) + 'type:{}\n'.format(self.type)
	
	def get_label_entity_type(self):
		arg_array = np.ones(len(self.sentence.tokens),)

		e1_token_start, e1_token_end = mention_to_token_span(self.e1, self.sentence)
		e2_token_start, e2_token_end = mention_to_token_span(self.e2, self.sentence)

		arg_array[e1_token_start:e1_token_end+1] = e_type2idx[self.e1.type]
		arg_array[e2_token_start:e2_token_end+1] = e_type2idx[self.e2.type]

		return arg_array
	
	def get_label_position(self):
		arg_array = np.ones(len(self.sentence.tokens),)

		e1_token_start, e1_token_end = mention_to_token_span(self.e1, self.sentence)
		e2_token_start, e2_token_end = mention_to_token_span(self.e2, self.sentence)

		arg_array[e1_token_start:e1_token_end+1] = 2
		arg_array[e2_token_start:e2_token_end+1] = 2

		return arg_array
	
	def get_type_label(self):
		return r_label2idx[self.type]
	
	def get_tokens(self):
		return [t.word for t in self.sentence.tokens]
	
	def get_word_idx(self, max_sent_len, word2idx):
		out_array = np.zeros((max_sent_len,))
		token_word_idx = get_idx_sequence(self.get_tokens(), word2idx)
		out_array[:len(token_word_idx)] = token_word_idx
		return out_array

