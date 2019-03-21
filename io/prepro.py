import tqdm
import os
import numpy as np
import ast
import word_embeddings
from io import io
import config

config = config.define_config()

module_location = os.path.abspath(__file__)
module_location = os.path.dirname(module_location)

word2vec = {}
# TODO : word2vec

with open(os.path.join(module_location, "../structure/", model_params["property2idx"])) as f:
	property2idx = ast.literal_eval(f.read())


def relation_to_indices(relations, word2idx):
	"""
	:param relations: relation graphs
	:param word2idx:
	:return:
		sentences_matrix[graph(sentence)_index, token_ids_in_sentence] = token_ids_in_word2idx
		entity_matrix[graph(sentence)_index, edge(relation)_index, token_ids_in_sentence] = entity_marker(1~4)('The', 4)
		y_matrix[graph(sentence)_index, edge(relation)_index] = relation_type (relation_type)
	"""
	max_sent_len = model_params['max_sent_len']  # 200
	sentences_matrix = np.zeros((len(relations), max_sent_len), dtype="int32")  # (sentence_number, sentence_len)
	arg1_matrix = np.zeros((len(relations), max_sent_len), dtype="int8")
	arg2_matrix = np.zeros((len(relations), max_sent_len), dtype="int8")
	y_matrix = np.zeros((len(relations), 1), dtype="int16")  # relation type 1~7

	for index, g in enumerate(tqdm.tqdm(relations, ascii=True)):
		token_wordvec_ids = word_embeddings.get_idx_sequence(g["Tokens"], word2idx)
		sentences_matrix[index, :len(token_wordvec_ids)] = token_wordvec_ids

		arg1_matrix[index, :len(token_wordvec_ids)] = 1
		arg2_matrix[index, :len(token_wordvec_ids)] = 1

		arg1_matrix[index, range(g["mentionArg1"]["start"], g["mentionArg1"]["end"] + 1)] = 2
		arg2_matrix[index, range(g["mentionArg2"]["start"], g["mentionArg2"]["end"] + 1)] = 2

		relation_type = g["relationType"]
		relation_type_id = property2idx.get(relation_type)
		y_matrix[index] = relation_type_id

	return sentences_matrix, arg1_matrix, arg2_matrix, y_matrix


def entity_to_indices(words, word2idx):
	return NotImplementedError


def tokens2embs(tokens):
	out_embs = np.zeros((config.max_sent_len, config.word_emb_dim))

	for index, t in enumerate(tokens):
		out_embs[index, :] = word2vec[t]
	return out_embs


def get_entity_type_from_id(entityid, entity_dir):
	entity_type = ''
	return entity_type


def get_entity_idx(r):
	label2idx = {'B-PER': 1, 'I-PER': 2, 'L-PER': 3, 'U-PER': 4,
				  'B-ORG': 5, 'I-ORG': 6, 'L-ORG': 7, 'U-ORG': 8,
				  'B-LOC': 9, 'I-LOC':10, 'L-LOC':11, 'U-LOC':12,
				  'B-GPE':13, 'I-GPE':14, 'L-GPE':15, 'U-GPE':16,
				  'B-FAC':17, 'I-FAC':18, 'L-FAC':19, 'U-FAC':20,
				  'B-VEH':21, 'I-VEH':22, 'L-VEH':23, 'U-VEH':24,
				  'B-WEA':25, 'I-WEA':26, 'L-WEA':27, 'U-WEA':28, 'O':29}

	out_idx = np.zeros((config.max_sent_len,))

	for i in range(len(r['Tokens'])):
		out_idx[i] = label2idx['O']

	with r['mentionArg1'] as e:
		for i in range(e['start'], e['end']):
			if e['end'] == e['start']:
				key = 'U'
			elif i == e['start']:
				key = 'B'
			elif i == e['end']:
				key = 'L'
			else:
				key = 'I'
			key = key + '-' + get_entity_type_from_id(e['argMentionid'])
			out_idx[i] = label2idx[key]

	with r['mentionArg2'] as e:
		for i in range(e['start'], e['end']):
			if e['end'] == e['start']:
				key = 'U'
			elif i == e['start']:
				key = 'B'
			elif i == e['end']:
				key = 'L'
			else:
				key = 'I'
			key = key + '-' + get_entity_type_from_id(e['argMentionid'])
			out_idx[i] = label2idx[key]

	return out_idx


def get_entity_data(relationMention_fp):
	train_data, val_data, test_data = io.load_relation_from_existing_sets(relationMention_fp)

	train_word_embs = np.zeros((len(train_data), config.max_sent_len, config.word_emb_dim))
	val_word_embs = np.zeros((len(val_data), config.max_sent_len, config.word_emb_dim))
	test_word_embs = np.zeros((len(test_data), config.max_sent_len, config.word_emb_dim))

	train_labels = np.zeros((len(train_data), config.max_sent_len))
	val_labels = np.zeros((len(val_data), config.max_sent_len))
	test_labels = np.zeros((len(test_data), config.max_sent_len))

	for index, r in enumerate(train_data):
		train_word_embs[index, :, :] = tokens2embs(r["Tokens"])
		train_labels[index, :] = get_entity_idx(r)
	for index, r in enumerate(val_data):
		val_word_embs[index, :] = tokens2embs(r["Tokens"])
		val_labels[index, :] = get_entity_idx(r)
	for index, r in enumerate(val_data):
		test_word_embs[index, :] = tokens2embs(r["Tokens"])
		test_labels[index, :] = get_entity_idx(r)

	return train_word_embs, train_labels, val_word_embs, val_labels, test_word_embs, test_labels


def get_multi_data():
	return NotImplementedError
