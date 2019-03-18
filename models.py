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


def multi_task_model():
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
	relation_labels = tf.placeholder(tf.float32, (None, relation_type_n))
	entity_idx_1 = tf.placeholder(tf.int8, (None, None))

	return predict_entity, entity_loss


def entity_model(batch_size=125):

	with tf.variable_scope('biLSTM1'):
		w_vec = tf.placeholder(tf.float32, (None, maxSent, w_vec_dim))
		w = tf.Variable(np.random.normal(), name='w')
		y_pred = tf.mul(w, x)
		w2h_layer = keras.layers.CuDNNLSTM(units=128, return_sequences=True, name='wordvec2h')
		h0_out = w2h_layer(w_vec)

		prediction = tf.nn.softmax()
		label = tf.placeholder(tf.float32, [batch_size, r_type_n], name='label')

		loss = -tf.reduce_sum(label * tf.log(prediction), axis=1)

	return loss, h0_out


def run_entity_model():
	init = tf.global_variables_initializer()

	w_vec = []

	loss, h0_out = entity_model()
	optimizer = tf.train.AdamOptimizer().minimize(loss)

	with tf.Session() as sess:
		sess.run(init)
		sess.run((loss, h0_out), feed_dict={'w_vec': w_vec, 'label': []})


if __name__ == '__main__':
	multi_task_model()

