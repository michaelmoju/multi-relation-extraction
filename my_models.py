import tensorflow as tf
import tensorflow.python.keras as keras
from tensorflow.python.keras import layers, Model

import word_embeddings
import config

train_m = 6000
r_type_n = 7
w_vec_dim = 50
biLSTM_hidd_n = 128
maxSent = 200
entity_type_n =29
relation_type_n = 6

config = config.define_config()


def multi_task_model(train_entity=False):
	sent_matrix = tf.placeholder(tf.float32, (None, maxSent, w_vec_dim))
	entity_labels = tf.placeholder(tf.float32, (None, maxSent, entity_type_n))
	print(entity_labels)

	entity_embeddings = tf.get_variable('entity_embeddings', [entity_type_n, embedding_size])

	embedded_entity_ids = tf.nn.embedding_lookup(entity_embeddings, entity_labels)

	# TODO: bi-LSTM
	w2h_layer = keras.layers.CuDNNLSTM(units=128, return_sequences=True, name='entity_biLSTM')
	h_bi = w2h_layer(sent_matrix)
	print(h_bi)
	entity_out = keras.layers.Dense(units=entity_type_n)(h_bi)
	predict_entity = tf.nn.softmax(entity_out)
	print(predict_entity)
	entity_loss = -tf.reduce_sum(entity_labels * tf.log(predict_entity), axis=1)
	print('entityloss:' + str(entity_loss))


	if train_entity:
		return predict_entity, entity_loss
	relation_labels = tf.placeholder(tf.float32, (None, relation_type_n))
	entity_idx_1 = tf.placeholder(tf.int8, (None, None))

	return predict_entity, entity_loss


def model_entity(embeddings, lstm_size = 128, entity_type_n=29, max_sent_len=config.max_sent_len, dropout=False):
	print('\nStart model_entity...')
	print('word_embedding_shape:{}'.format(embeddings.shape))

	sentence_input = layers.Input((max_sent_len, ), dtype='int32', name='sentence_input')
	word_embeddings = layers.Embedding(output_dim=embeddings.shape[1],
										   input_dim=embeddings.shape[0],
										   weights=embeddings,
										   mask_zero=True,
										   input_length=max_sent_len, trainable=False, name='embedding_layer')(sentence_input)

	if dropout:
		word_embeddings = layers.Dropout(0.5)(word_embeddings)

	lstm_out = layers.Bidirectional(layers.LSTM(lstm_size, return_sequences=True, name='LSTM_layer'), name='BiLSTM_layer')(word_embeddings)

	if dropout:
		lstm_out = layers.Dropout(0.5)(lstm_out)

	main_out = layers.Dense(entity_type_n, activation='softmax', name='softmax_layer')(lstm_out)

	model = Model(inputs=sentence_input, outputs=main_out)
	model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

	return model


if __name__ == '__main__':

	embeddings, word2idx= word_embeddings.load_word_emb('../resource/embeddings/glove/glove.6B.50d.txt')
	model = model_entity(embeddings)

