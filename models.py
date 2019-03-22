import numpy as np
import tensorflow as tf
from tensorflow import keras

train_m = 6000
r_type_n = 7
w_vec_dim = 50
biLSTM_hidd_n = 128
maxSent = 200
entity_type_n = 7
relation_type_n = 6


def multi_task_model(train_entity=False):
	sent_matrix = tf.placeholder(tf.float32, (None, maxSent, w_vec_dim))
	entity_labels = tf.placeholder(tf.float32, (None, maxSent, entity_type_n))
	print(entity_labels)

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


def entity_model(batch_size=125):

	with tf.variable_scope('biLSTM1'):
		sent_matrix = tf.placeholder(tf.float32, (None, maxSent, w_vec_dim))
		entity_labels = tf.placeholder(tf.float32, (None, maxSent, entity_type_n))

		w2h_layer_f = tf.keras.layers.CuDNNLSTM(units=128, return_sequences=True, name='entity_biLSTM_f')
		w2h_layer_b = tf.keras.layers.CuDNNLSTM(units=128, return_sequences=True, go_backwards=True, name='entity_biLSTM_b')

		dense_layer = tf.keras.layers.Dense(units=entity_type_n, name='entity_dense')

		loss = 0

		for word_vec_f, word_vec_b in w2h_layer_f(sent_matrix), w2h_layer_b(sent_matrix):
			w2h_out = tf.concat(word_vec_f, word_vec_b)
			dense_out = dense_layer(w2h_out)
			entity_type_out = tf.nn.softmax(dense_out)
			loss += -tf.reduce_sum(entity_labels * tf.log(entity_type_out), axis=1)

		# TODO: return the sequence of labels

	return loss, entity_type_out


def run_entity_model():
	init = tf.global_variables_initializer()

	tokens = []

	loss, h0_out = entity_model()
	optimizer = tf.train.AdamOptimizer().minimize(loss)

	feed_dic = {'w_vec': tokens, 'label': []}

	with tf.Session() as sess:
		sess.run(init)
		sess.run((loss, h0_out, optimizer), feed_dict=feed_dic)


if __name__ == '__main__':
	multi_task_model()

