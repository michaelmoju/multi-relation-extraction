import os
import json
import tensorflow as tf
from tensorflow.keras import layers, Model, utils
import numpy as np

import word_embeddings
import config

module_location = os.path.abspath(__file__)
module_location = os.path.dirname(module_location)

with open(os.path.join(module_location, "./model_params.json")) as f:
	p = json.load(f)

e_label2idx = {'B-PER': 1, 'I-PER': 2, 'L-PER': 3, 'U-PER': 4,
				 'B-ORG': 5, 'I-ORG': 6, 'L-ORG': 7, 'U-ORG': 8,
				 'B-LOC': 9, 'I-LOC': 10, 'L-LOC': 11, 'U-LOC': 12,
				 'B-GPE': 13, 'I-GPE': 14, 'L-GPE': 15, 'U-GPE': 16,
				 'B-FAC': 17, 'I-FAC': 18, 'L-FAC': 19, 'U-FAC': 20,
				 'B-VEH': 21, 'I-VEH': 22, 'L-VEH': 23, 'U-VEH': 24,
				 'B-WEA': 25, 'I-WEA': 26, 'L-WEA': 27, 'U-WEA': 28, 'O': 0}

r_label2idx = {'PHYS': 1, 'PART-WHOLE': 2, 'PER-SOC': 3, 'ORG-AFF': 4, 'ART': 5, 'GEN-AFF': 6, 'METONYMY': 7}


def model_relation(embeddings, entity_weights, train_entity=False, dropout=False):
	print('\nStart model_relation... Train_entity:{}'.format(train_entity))
	print('word_embedding_shape:{}'.format(embeddings.shape))

	sentence_input = layers.Input((p['max_sent_len'],), dtype='int32', name='sentence_input')
	word_embeddings = layers.Embedding(embeddings.shape[0], embeddings.shape[1],
									   weights=embeddings,
									   input_length=p['max_sent_len'],
									   trainable=False,
									   mask_zero=True,
									   name='embedding_layer')(sentence_input)
	print(word_embeddings.shape)
	if dropout:
		word_embeddings = layers.Dropout(0.5)(word_embeddings)

	if train_entity:
		lstm1_out = layers.Bidirectional(layers.LSTM(p['lstm1'], return_sequences=True, name='entity_LSTM_layer'),
										 name='entity_BiLSTM_layer')(word_embeddings)
	else:
		lstm1_out = layers.Bidirectional(
			layers.LSTM(p['lstm1'], return_sequences=True, trainable=False, name='entity_LSTM_layer'), weights=entity_weights, name='entity_BiLSTM_layer')(word_embeddings)
	if dropout:
		lstm1_out = layers.Dropout(0.5)(lstm1_out)

	entity_class = layers.Dense(p['entity_type_n'], activation='softmax', name='entity_softmax_layer')(lstm1_out)

	# Take arg1_markers that identify entity positions, convert to position embeddings
	arg1_markers = layers.Input((p['max_sent_len'],), dtype='int8', name='arg1_markers')
	arg1_pos_embeddings = layers.Embedding(p['entity_type_n'], p['position_emb'],
										   input_length=p['max_sent_len'],
										   mask_zero=True, trainable=True)(arg1_markers)

	# Take arg2_markers that identify entity positions, convert to position embeddings
	arg2_markers = layers.Input((p['max_sent_len'],), dtype='int8', name='arg2_markers')
	arg2_pos_embeddings = layers.Embedding(p['entity_type_n'], p['position_emb'],
										   input_length=p['max_sent_len'],
										   mask_zero=True, trainable=True)(arg2_markers)

	concate = layers.concatenate([lstm1_out, arg1_pos_embeddings, arg2_pos_embeddings])
	lstm2_out = layers.LSTM(p['lstm2'], name='relation_LSTM_layer')(concate)
	# TODO: dropout
	main_out = layers.Dense(p['relation_type_n'], activation='softmax', name='relation_softmax_layer')(lstm2_out)

	model = Model(inputs=[sentence_input, arg1_markers, arg2_markers], outputs=main_out)
	adamopt = tf.optimizers.Adam(p['learning_rate'])
	model.compile(optimizer=adamopt, loss='categorical_crossentropy', metrics=['accuracy'])

	return model


def model_entity(embeddings, dropout=False):
	print('\nStart model_entity...')
	print('word_embedding_shape:{}'.format(embeddings.shape))

	sentence_input = layers.Input((p['max_sent_len'],), dtype='int32', name='sentence_input')
	word_embeddings = layers.Embedding(embeddings.shape[0], embeddings.shape[1],
									   weights = embeddings,
									   input_length=p['max_sent_len'],
									   trainable=False,
									   mask_zero=True,
									   name='embedding_layer')(sentence_input)
	if dropout:
		word_embeddings = layers.Dropout(0.5)(word_embeddings)

	lstm_out = layers.Bidirectional(layers.LSTM(p['lstm1'], return_sequences=True, name='entity_LSTM_layer'), name='entity_BiLSTM_layer')(word_embeddings)
	if dropout:
		lstm_out = layers.Dropout(0.5)(lstm_out)

	main_out = layers.Dense(p['entity_type_n'], activation='softmax', name='entity_softmax_layer')(lstm_out)

	model = Model(inputs=sentence_input, outputs=main_out)
	adamopt = tf.optimizers.Adam(p['learning_rate'])
	model.compile(optimizer=adamopt, loss='categorical_crossentropy', metrics=['accuracy'])

	return model


def to_indices_with_entity_instances(instances, word2idx):
	out_sent_matrix = np.zeros((len(instances), p['max_sent_len']), dtype="int32")
	label_matrix = np.zeros((len(instances), p['max_sent_len']), dtype="int8")

	for index, instance in enumerate(instances):
		out_sent_matrix[index, :] = instance.get_word_idx(p['max_sent_len'], word2idx)
		label_matrix[index, :] = instance.get_label(e_label2idx, p['max_sent_len'])

	return out_sent_matrix, label_matrix


def to_indices_with_extracted_entities(relation_instances, word2idx):
	max_sent_len = p['max_sent_len']  # 120
	sentences_matrix = np.zeros((len(relation_instances), max_sent_len), dtype="int32")  # (sentence_number, sentence_len)
	arg1_matrix = np.zeros((len(relation_instances), max_sent_len), dtype="int8")
	arg2_matrix = np.zeros((len(relation_instances), max_sent_len), dtype="int8")
	y_matrix = np.zeros((len(relation_instances), 1), dtype="int16")  # relation type 1~7

	for index, r in enumerate(relation_instances):

		token_wordvec_ids = word_embeddings.get_idx_sequence(r["Tokens"], word2idx)
		sentences_matrix[index, :len(token_wordvec_ids)] = token_wordvec_ids

		arg1_matrix[index, :len(token_wordvec_ids)] = 1
		arg2_matrix[index, :len(token_wordvec_ids)] = 1

		arg1_matrix[index, range(r["mentionArg1"]["start"], r["mentionArg1"]["end"]+1)] = 2
		arg2_matrix[index, range(r["mentionArg2"]["start"], r["mentionArg2"]["end"]+1)] = 2

		relation_type = r["relationType"]
		relation_type_id = r_label2idx.get(relation_type)
		y_matrix[index] = relation_type_id

	return sentences_matrix, arg1_matrix, arg2_matrix, y_matrix


if __name__ == '__main__':

	embeddings, word2idx= word_embeddings.load_word_emb('../resource/embeddings/glove/glove.6B.50d.txt')
	model = model_entity(embeddings)
	print(model.summary())
	utils.plot_model(model, './trainedmodels/entity_model.png', show_shapes=True)

	entity_weights = model.get_layer(name='entity_BiLSTM_layer').get_weights()
	print('embedding_layer dtype:{}'.format(model.get_layer(name='embedding_layer').dtype))
	model = model_relation(embeddings, entity_weights)
	print(model.summary())
	utils.plot_model(model, './trainedmodels/relation_model.png', show_shapes=True)


