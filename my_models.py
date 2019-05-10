import os
import json
from miwaIO.instance import e_type2idx
from tensorflow.python.keras import layers, Model, utils, regularizers
import numpy as np
import word_embeddings

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
e_idx2label = {v: k for k, v in e_label2idx.items()}

r_label2idx = {'PHYS': 1, 'PART-WHOLE': 2, 'PER-SOC': 3, 'ORG-AFF': 4, 'ART': 5, 'GEN-AFF': 6}
r_idx2label = {v: k for k, v in r_label2idx.items()}

POSITION_VOCAB_SIZE = 3


def load_eType_embeddings():
	embeddings = np.zeros((9, 8), dtype='float32')
	
	for i in range(1, 9):
		embeddings[i, i - 1] = 1
	return embeddings


def model_relation_LSTMbaseline(embeddings):
	print('\nStart model_relation_LSTMbaseline...')
	print('word_embedding_shape:{}'.format(embeddings.shape))
	
	sentence_input = layers.Input((p['max_sent_len'],), dtype='int32', name='sentence_input')
	word_embeddings = layers.Embedding(embeddings.shape[0], embeddings.shape[1],
	                                   weights=[embeddings],
	                                   input_length=p['max_sent_len'],
	                                   trainable=False,
	                                   mask_zero=True,
	                                   name='embedding_layer')(sentence_input)
	
	word_embeddings = layers.Dropout(p['dropout'])(word_embeddings)
	
	# Take arg1_markers that identify entity positions, convert to position embeddings
	arg_markers = layers.Input((p['max_sent_len'],), dtype='int8', name='arg1_markers')
	arg_pos_embeddings = layers.Embedding(POSITION_VOCAB_SIZE, p['position_emb'],
	                                       input_length=p['max_sent_len'],
	                                       mask_zero=True,
	                                       embeddings_regularizer=regularizers.l2(),
	                                       trainable=True)(arg_markers)
	
	concate = layers.concatenate([word_embeddings, arg_pos_embeddings])
	lstm2_out = layers.LSTM(p['lstm2'], name='relation_LSTM_layer')(concate)
	lstm2_out = layers.Dropout(p['dropout'])(lstm2_out)
	main_out = layers.Dense(p['relation_type_n'], activation='softmax', name='relation_softmax_layer')(lstm2_out)
	model = Model(inputs=[sentence_input, arg_markers], outputs=[main_out])
	
	return model


def model_relation_LSTMtype(embeddings, eType_embeddings):
	print('\nStart model_relation_LSTMbaseline...')
	print('word_embedding_shape:{}'.format(embeddings.shape))
	
	sentence_input = layers.Input((p['max_sent_len'],), dtype='int32', name='sentence_input')
	word_embeddings = layers.Embedding(embeddings.shape[0], embeddings.shape[1],
	                                   weights=[embeddings],
	                                   input_length=p['max_sent_len'],
	                                   trainable=False,
	                                   mask_zero=True,
	                                   embeddings_regularizer=regularizers.l2(),
	                                   name='embedding_layer')(sentence_input)
	
	word_embeddings = layers.Dropout(p['dropout'])(word_embeddings)
	
	# Take arg1_markers that identify entity positions, convert to position embeddings
	arg_markers = layers.Input((p['max_sent_len'],), dtype='int8', name='arg1_markers')
	arg_pos_embeddings = layers.Embedding(eType_embeddings.shape[0], eType_embeddings.shape[1],
	                                      weights=[eType_embeddings],
	                                      input_length=p['max_sent_len'],
	                                      mask_zero=True,
	                                      trainable=False)(arg_markers)
	
	concate = layers.concatenate([word_embeddings, arg_pos_embeddings])
	lstm2_out = layers.LSTM(p['lstm2'], name='relation_LSTM_layer')(concate)
	lstm2_out = layers.Dropout(p['dropout'])(lstm2_out)
	main_out = layers.Dense(p['relation_type_n'], activation='softmax', name='relation_softmax_layer')(lstm2_out)
	model = Model(inputs=[sentence_input, arg_markers], outputs=[main_out])
	
	return model


def model_relation_entity_LSTM(embeddings, entity_weights, train_entity=False, dropout=False):
	print('\nStart model_relation_entity_LSTM...')
	print('word_embedding_shape:{}'.format(embeddings.shape))
	
	sentence_input = layers.Input((p['max_sent_len'],), dtype='int32', name='sentence_input')
	word_embeddings = layers.Embedding(embeddings.shape[0], embeddings.shape[1],
	                                   weights=[embeddings],
	                                   input_length=p['max_sent_len'],
	                                   trainable=False,
	                                   mask_zero=True,
	                                   name='embedding_layer')(sentence_input)
	
	if train_entity:
		lstm1_out = layers.Bidirectional(
			layers.LSTM(p['lstm1'], return_sequences=True, trainable=False, name='entity_LSTM_layer'),
			weights=entity_weights, name='entity_BiLSTM_layer')(word_embeddings)
		arg1_input = layers.Input((p['max_sent_len'], p['lstm2'] * 2), dtype='float32', name='arg1_input')
		arg2_input = layers.Input((p['max_sent_len'], p['lstm2'] * 2), dtype='float32', name='arg2_input')
	else:
		lstm1_out = word_embeddings
		arg1_input = layers.Input((p['max_sent_len'], p['word_emb']), dtype='float32', name='arg1_indicate')
		arg2_input = layers.Input((p['max_sent_len'], p['word_emb']), dtype='float32', name='arg2_indicate')
	
	if dropout:
		lstm1_out = layers.Dropout(p['dropout'])(lstm1_out)
	
	entity_class = layers.Dense(p['entity_type_n'], activation='softmax', name='entity_softmax_layer')(lstm1_out)
	
	arg_lstm = layers.LSTM(p['lstm2'])
	
	arg1 = layers.Multiply()([arg1_input, lstm1_out])
	arg1 = layers.Masking(mask_value=0)(arg1)
	arg1_out = arg_lstm(arg1)
	
	arg2 = layers.Multiply()([arg2_input, lstm1_out])
	arg2 = layers.Masking(mask_value=0)(arg2)
	arg2_out = arg_lstm(arg2)
	
	arg_out = layers.concatenate([arg1_out, arg2_out])
	main_out = layers.Dense(p['relation_type_n'], activation='softmax', name='relation_softmax_layer')(arg_out)
	model = Model(inputs=[sentence_input, arg1_input, arg2_input], outputs=[main_out])
	
	return model


def model_relation_multi(embeddings, entity_weights, train_entity=False, dropout=False):
	print('\nStart model_relation... Train_entity:{}'.format(train_entity))
	print('word_embedding_shape:{}'.format(embeddings.shape))
	
	sentence_input = layers.Input((p['max_sent_len'],), dtype='int32', name='sentence_input')
	word_embeddings = layers.Embedding(embeddings.shape[0], embeddings.shape[1],
	                                   weights=[embeddings],
	                                   input_length=p['max_sent_len'],
	                                   trainable=False,
	                                   mask_zero=True,
	                                   name='embedding_layer')(sentence_input)
	print(word_embeddings.shape)
	if dropout:
		word_embeddings = layers.Dropout(p['dropout'])(word_embeddings)
	
	if train_entity:
		lstm1_out = layers.Bidirectional(layers.LSTM(p['lstm1'], return_sequences=True, name='entity_LSTM_layer'),
		                                 name='entity_BiLSTM_layer')(word_embeddings)
	else:
		lstm1_out = layers.Bidirectional(
			layers.LSTM(p['lstm1'], return_sequences=True, trainable=False, name='entity_LSTM_layer'),
			weights=entity_weights, name='entity_BiLSTM_layer')(word_embeddings)
	if dropout:
		lstm1_out = layers.Dropout(p['dropout'])(lstm1_out)
	
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
	
	model = Model(inputs=[sentence_input, arg1_markers, arg2_markers], outputs=[main_out])
	
	return model


def model_entity(embeddings, dropout=False):
	print('\nStart model_entity...')
	print('word_embedding_shape:{}'.format(embeddings.shape))
	
	sentence_input = layers.Input((p['max_sent_len'],), dtype='int32', name='sentence_input')
	word_embeddings = layers.Embedding(embeddings.shape[0], embeddings.shape[1],
	                                   weights=[embeddings],
	                                   input_length=p['max_sent_len'],
	                                   trainable=False,
	                                   mask_zero=True,
	                                   name='embedding_layer')(sentence_input)
	if dropout:
		word_embeddings = layers.Dropout(p['dropout'])(word_embeddings)
	
	lstm_out = layers.Bidirectional(layers.LSTM(p['lstm1'], return_sequences=True, name='entity_LSTM_layer'),
	                                name='entity_BiLSTM_layer')(word_embeddings)
	if dropout:
		lstm_out = layers.Dropout(p['dropout'])(lstm_out)
	
	main_out = layers.Dense(p['entity_type_n'], activation='softmax', name='entity_softmax_layer')(lstm_out)
	
	model = Model(inputs=sentence_input, outputs=main_out)
	
	return model


def to_indices_with_entity_instances(instances, word2idx):
	out_sent_matrix = np.zeros((len(instances), p['max_sent_len']), dtype="int32")
	label_matrix = np.zeros((len(instances), p['max_sent_len']), dtype="int8")
	
	for index, instance in enumerate(instances):
		out_sent_matrix[index, :] = instance.get_word_idx(p['max_sent_len'], word2idx)
		label_matrix[index, :] = instance.get_label(e_label2idx, p['max_sent_len'])
	
	return out_sent_matrix, label_matrix


def r_to_indices_type_e(instances, word2idx):
	pass


def r_to_indices_position_e(instances, word2idx):
	max_sent_len = p['max_sent_len']  # 120
	sentences_matrix = np.zeros((len(instances), max_sent_len), dtype="int32")  # (sentence_number, sentence_len)
	arg_matrix = np.zeros((len(instances), max_sent_len), dtype="int8")
	y_matrix = np.zeros((len(instances), 1), dtype="int16")
	
	for index, instance in enumerate(instances):
		sentences_matrix[index, :] = instance.get_word_idx(p['max_sent_len'], word2idx)
		
		arg_matrix[index, :len(instance.get_tokens())] = instance.get_label_position()
		
		y_matrix[index] = instance.get_type_label()
	
	return sentences_matrix, arg_matrix, y_matrix


def r_to_indices_typed_e(instances, word2idx):
	max_sent_len = p['max_sent_len']  # 120
	sentences_matrix = np.zeros((len(instances), max_sent_len), dtype="int32")  # (sentence_number, sentence_len)
	arg_matrix = np.zeros((len(instances), max_sent_len), dtype="int8")
	y_matrix = np.zeros((len(instances), 1), dtype="int16")
	
	for index, instance in enumerate(instances):
		sentences_matrix[index, :] = instance.get_word_idx(p['max_sent_len'], word2idx)
		
		arg_matrix[index, :len(instance.get_tokens())] = instance.get_label_position()
		
		y_matrix[index] = instance.get_type_label()
	
	return sentences_matrix, arg_matrix, y_matrix


def r_to_indices_e_mat(instances, word2idx):
	max_sent_len = p['max_sent_len']  # 120
	sentences_matrix = np.zeros((len(instances), max_sent_len), dtype="int32")  # (sentence_number, sentence_len)
	arg1_matrix = np.zeros((len(instances), max_sent_len, p['word_emb']), dtype="int8")
	arg2_matrix = np.zeros((len(instances), max_sent_len, p['word_emb']), dtype="int8")
	y_matrix = np.zeros((len(instances), 1), dtype="int16")  # relation type 1~7
	
	for index, instance in enumerate(instances):
		sentences_matrix[index, :] = instance.get_word_idx(p['max_sent_len'], word2idx)
		
		arg1_matrix[index, :, :], arg2_matrix[index, :, :] = instance.get_one_hot_position(p, train_entity=False)
		
		y_matrix[index] = r_label2idx[instance.type]
	
	return sentences_matrix, arg1_matrix, arg2_matrix, y_matrix


def r_to_indices_e_mat_train_entity(instances, word2idx):
	max_sent_len = p['max_sent_len']  # 120
	sentences_matrix = np.zeros((len(instances), max_sent_len), dtype="int32")  # (sentence_number, sentence_len)
	arg1_matrix = np.zeros((len(instances), max_sent_len, p['lstm2'] * 2), dtype="int8")
	arg2_matrix = np.zeros((len(instances), max_sent_len, p['lstm2'] * 2), dtype="int8")
	
	y_matrix = np.zeros((len(instances), 1), dtype="int16")  # relation type 1~7
	
	for index, instance in enumerate(instances):
		sentences_matrix[index, :] = instance.get_word_idx(p['max_sent_len'], word2idx)
		
		arg1_matrix[index, :, :], arg2_matrix[index, :, :] = instance.get_one_hot_position(p, train_entity=True)
		
		y_matrix[index] = r_label2idx.get(instance.type)
	
	return sentences_matrix, arg1_matrix, arg2_matrix, y_matrix


if __name__ == '__main__':
	embeddings, word2idx = word_embeddings.load_word_emb('../resource/embeddings/glove/glove.6B.50d.txt')
	# model = model_entity(embeddings, dropout=True)
	# print(model.summary())
	# utils.plot_model(model, './trainedmodels/entity_model.png', show_shapes=True)
	#
	# entity_weights = model.get_layer(name='entity_BiLSTM_layer').get_weights()
	# # print('embedding_layer dtype:{}'.format(model.get_layer(name='embedding_layer').dtype))
	# # model = model_relation_LSTMbaseline(embeddings)
	# # print(model.summary())
	# # utils.plot_model(model, './trainedmodels/relation_model.png', show_shapes=True)
	#
	# print(model.summary())
	# utils.plot_model(model, './trainedmodels/relation_model1.png', show_shapes=True)
	
	eType_embeddings = load_eType_embeddings()
	model = model_relation_LSTMtype(embeddings, eType_embeddings)
	print(model.summary())
	utils.plot_model(model, './trainedmodels/relation_model1.png', show_shapes=True)