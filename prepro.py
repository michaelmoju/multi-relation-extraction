import tqdm
import os
import json
import ast
import word_embeddings

module_location = os.path.abspath(__file__)
module_location = os.path.dirname(module_location)

with open(os.path.join(module_location, "../model_params.json")) as f:
	model_params = json.load(f)

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

		arg1_matrix[index, range(g["mentionArg1"]["start"], g["mentionArg1"]["end"]+1)] = 2
		arg2_matrix[index, range(g["mentionArg2"]["start"], g["mentionArg2"]["end"]+1)] = 2

		relation_type = g["relationType"]
		relation_type_id = property2idx.get(relation_type)
		y_matrix[index] = relation_type_id

	return sentences_matrix, arg1_matrix, arg2_matrix, y_matrix


def entity_to_indices(words, word2idx):
	return NotImplementedError