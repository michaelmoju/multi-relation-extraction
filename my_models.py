import os
import json
import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np

import word_embeddings
import config

module_location = os.path.abspath(__file__)
module_location = os.path.dirname(module_location)

with open(os.path.join(module_location, "./model_params.json")) as f:
	model_params = json.load(f)

label2idx = {'B-PER': 1, 'I-PER': 2, 'L-PER': 3, 'U-PER': 4,
				 'B-ORG': 5, 'I-ORG': 6, 'L-ORG': 7, 'U-ORG': 8,
				 'B-LOC': 9, 'I-LOC': 10, 'L-LOC': 11, 'U-LOC': 12,
				 'B-GPE': 13, 'I-GPE': 14, 'L-GPE': 15, 'U-GPE': 16,
				 'B-FAC': 17, 'I-FAC': 18, 'L-FAC': 19, 'U-FAC': 20,
				 'B-VEH': 21, 'I-VEH': 22, 'L-VEH': 23, 'U-VEH': 24,
				 'B-WEA': 25, 'I-WEA': 26, 'L-WEA': 27, 'U-WEA': 28, 'O': 0}


def multi_task_model(train_entity=False):
	pass


def model_entity(embeddings, lstm_size = 128, dropout=False):
	print('\nStart model_entity...')
	print('word_embedding_shape:{}'.format(embeddings.shape))

	sentence_input = layers.Input((model_params['max_sent_len'], ), dtype='int32', name='sentence_input')
	word_embeddings = layers.Embedding(embeddings.shape[0], embeddings.shape[1],
									   weights = embeddings,
									   input_length=model_params['max_sent_len'],
									   trainable=False,
									   mask_zero=True,
									   name='embedding_layer')(sentence_input)
	print(word_embeddings.shape)
	if dropout:
		word_embeddings = layers.Dropout(0.5)(word_embeddings)

	lstm_out = layers.Bidirectional(layers.LSTM(lstm_size, return_sequences=True, name='LSTM_layer'), name='BiLSTM_layer')(word_embeddings)

	print(lstm_out.shape)

	if dropout:
		lstm_out = layers.Dropout(0.5)(lstm_out)

	main_out = layers.Dense(model_params['entity_type_n'], activation='softmax', name='softmax_layer')(lstm_out)
	print(main_out.shape)

	model = Model(inputs=sentence_input, outputs=main_out)
	model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

	return model


def to_indices_with_entity_instances(instances, word2idx):
	out_sent_matrix = np.zeros((len(instances), model_params['max_sent_len']), dtype="int32")
	label_matrix = np.zeros((len(instances), model_params['max_sent_len']), dtype="int8")

	for index, instance in enumerate(instances):
		out_sent_matrix[index, :] = instance.get_word_idx(model_params['max_sent_len'], word2idx)
		label_matrix[index, :] = instance.get_label(label2idx, model_params['max_sent_len'])

	return out_sent_matrix, label_matrix


if __name__ == '__main__':

	embeddings, word2idx= word_embeddings.load_word_emb('../resource/embeddings/glove/glove.6B.50d.txt')
	model = model_entity(embeddings)
	print(model.summary())

