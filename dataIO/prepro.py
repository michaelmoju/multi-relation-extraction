# Author: Moju Wu

import json
import re
import os
import numpy as np
import word_embeddings
from dataIO import io
from dataIO.annotation import *

config = config.define_config()

module_location = os.path.abspath(__file__)
module_location = os.path.dirname(module_location)


# with open(os.path.join(module_location, "../model_params.json")) as f:
# 	model_params = json.load(f)
#
# with open(os.path.join(module_location, "../structure/", model_params["property2idx"])) as f:
# 	property2idx = ast.literal_eval(f.read())


def relation_to_indices(relations, word2idx):
	"""
	:param relations: relation graphs
	:param word2idx:
	:return:
		sentences_matrix[graph(sentence)_index, token_ids_in_sentence] = token_ids_in_word2idx
		entity_matrix[graph(sentence)_index, edge(relation)_index, token_ids_in_sentence] = entity_marker(1~4)('The', 4)
		y_matrix[graph(sentence)_index, edge(relation)_index] = relation_type (relation_type)
	"""
	max_sent_len = config.max_sent_len
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
	pass


def tokens_to_embs(tokens, embeddings, word2idx):
	# out_embs = tf.nn.embedding_lookup(embeddings, [word2idx[t] for t in tokens])

	out_embs = np.zeros((config.max_sent_len, config.word_emb_dim), dtype='float32')

	for index, t in enumerate(tokens):
		emb = embeddings[word_embeddings.get_idx(t, word2idx)]
		out_embs[index, :] = emb

	return out_embs


def get_entity_type_from_id(entityid, entity_dir):
	print(entityid)

	# TODO: revise this with re
	*docID_ls, eID, mID = entityid.split('-')

	docID = ''
	for i in docID_ls:
		docID += i + '-'

	docID = docID[:-1]

	print(docID)

	with open(entity_dir + docID + '.apf.xml.entity.json') as f:
		entity_dic = json.load(f)
		entity_type = entity_dic[docID + '-' + eID]['entityType']

	return entity_type


def get_entity_idx(r, entity_dir):
	label2idx = {'B-PER': 1, 'I-PER': 2, 'L-PER': 3, 'U-PER': 4,
				 'B-ORG': 5, 'I-ORG': 6, 'L-ORG': 7, 'U-ORG': 8,
				 'B-LOC': 9, 'I-LOC': 10, 'L-LOC': 11, 'U-LOC': 12,
				 'B-GPE': 13, 'I-GPE': 14, 'L-GPE': 15, 'U-GPE': 16,
				 'B-FAC': 17, 'I-FAC': 18, 'L-FAC': 19, 'U-FAC': 20,
				 'B-VEH': 21, 'I-VEH': 22, 'L-VEH': 23, 'U-VEH': 24,
				 'B-WEA': 25, 'I-WEA': 26, 'L-WEA': 27, 'U-WEA': 28, 'O': 29}

	mention1_idx = np.zeros((config.max_sent_len,))
	mention2_idx = np.zeros((config.max_sent_len,))

	for i in range(len(r['Tokens'])):
		mention1_idx[i] = label2idx['O']
		mention2_idx[i] = label2idx['O']
	e = r['mentionArg1']
	for i in range(e['start'], e['end'] + 1):
		if e['end'] == e['start']:
			key = 'U'
		elif i == e['start']:
			key = 'B'
		elif i == e['end']:
			key = 'L'
		else:
			key = 'I'
		key = key + '-' + get_entity_type_from_id(e['argMentionid'], entity_dir)
		mention1_idx[i] = label2idx[key]

	e2 = r['mentionArg2']
	for i in range(e2['start'], e2['end'] + 1):
		if e2['end'] == e2['start']:
			key = 'U'
		elif i == e2['start']:
			key = 'B'
		elif i == e2['end']:
			key = 'L'
		else:
			key = 'I'
		key = key + '-' + get_entity_type_from_id(e2['argMentionid'], entity_dir)
		mention2_idx[i] = label2idx[key]

	return mention1_idx, mention2_idx


def load_entity_instances(entity_dir):
	entity_dicts = io.get_entity_from_files(entity_dir)
	entities = []
	entity_instances = {}
	nested_sentence_set = set()
	for entity_id, entity_dict in entity_dicts.items():
		entities.append(Entity().set_from_entity_dict(entity_dict))

	print(entities)
	for entity in entities:
		rgx = re.compile('(.*)-(E[0-9]+)')
		docID, _ = rgx.match(entity.id).groups()
		for mention in entity.mentions:
			sentenceID = docID + '-' + str(mention.sentence_index)
			if sentenceID not in entity_instances.keys():
				entity_instances[sentenceID] = EntityInstance(sentenceID, len(mention.tokens))
			else:
				entity_instances[sentenceID].add_mention(mention)
				if entity_instances[sentenceID].is_nested:
					print('nested entity mention found in {}'.format(sentenceID))
					nested_sentence_set.add(sentenceID)
	return entity_instances, nested_sentence_set


def get_entity_data_old(relationMention_fp, word_emb_fp, entity_dir):
	train_data, val_data, test_data = io.load_relation_from_existing_sets(relationMention_fp)
	embeddings, word2idx = word_embeddings.load_word_emb(word_emb_fp)

	train_word_embs = np.zeros((len(train_data), config.max_sent_len, config.word_emb_dim), dtype='float32')
	val_word_embs = np.zeros((len(val_data), config.max_sent_len, config.word_emb_dim), dtype='float32')
	test_word_embs = np.zeros((len(test_data), config.max_sent_len, config.word_emb_dim), dtype='float32')

	train_labels = np.zeros((len(train_data), 2, config.max_sent_len), dtype='int8')
	val_labels = np.zeros((len(val_data), 2, config.max_sent_len), dtype='int8')
	test_labels = np.zeros((len(test_data), 2, config.max_sent_len), dtype='int8')

	for index, r in enumerate(train_data):
		train_word_embs[index, :, :] = tokens_to_embs(r["Tokens"], embeddings, word2idx)
		train_labels[index, 0, :], train_labels[index, 1, :] = get_entity_idx(r, entity_dir)
	for index, r in enumerate(val_data):
		val_word_embs[index, :] = tokens_to_embs(r["Tokens"], embeddings, word2idx)
		val_labels[index, 0, :], val_labels[index, 1, :] = get_entity_idx(r, entity_dir)
	for index, r in enumerate(val_data):
		test_word_embs[index, :] = tokens_to_embs(r["Tokens"], embeddings, word2idx)
		test_labels[index, 0, :], test_labels[index, 1, :] = get_entity_idx(r, entity_dir)

	return train_word_embs, train_labels, val_word_embs, val_labels, test_word_embs, test_labels


def get_multi_data():
	pass


if __name__ == '__main__':
	# embeddings, word2idx = word_embeddings.load_word_emb('../../resource/embeddings/glove/glove.6B.50d.txt')
	# tokens = ['the', ',']
	# out_embs = tokens_to_embs(tokens, embeddings, word2idx)
	#
	# print(out_embs.shape)
	# print(out_embs[0,:])
	# print(out_embs[1,:])

	# --------------------------------------------------------------------

	# type = get_entity_type_from_id("CNN_CF_20030303.1900.00-E80-155", '../../resource/data/ace-2005/entityDictionary/adj_all/')
	# print(type)

	# --------------------------------------------------------------------

	# train_word_embs, train_labels, val_word_embs, val_labels, test_word_embs, test_labels = get_entity_data_old(
	# 	'../../resource/data/ace-2005/relationMention/data-set/', '../../resource/embeddings/glove/glove.6B.50d.txt',
	# 	'../../resource/data/ace-2005/entityDictionary/adj_all/')
	# print(train_labels[0, 0, :])
	# print(train_labels[0, 1, :])

	load_entity_instances('../../resource/data/ace-2005/entityDictionary/adj_all/')

