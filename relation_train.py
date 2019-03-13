import numpy as np
import tensorflow as tf

train_m = 6000
r_type_n = 7
w_vec_dim = 50


def entity_extraction(batch_size=125):
	w_vec = tf.placeholder(tf.float32, (w_vec_dim,))
	bi_lstm = tf.nn.bidirectional_dynamic_rnn()

	with tf.variable_scope('eext') as scope:
		w = tf.Variable(np.random.normal(), name='w')
		y_pred = tf.mul(w, x)

	prediction = tf.nn.softmax()
	label = tf.placeholder(tf.float32, [batch_size, r_type_n])

	cross_entropy = -tf.reduce_sum(label * tf.log(prediction), axis=1)

	train_step = tf.train.AdamOptimizer().minimize(cross_entropy)


	for i in range(1000):
		batch_x, batch_label = data.next_batch()
		sess.run(train_step, {w_vec: batch_x, label: batch_label})

if __name__ == '__main__':
	import argparse

	parser = argparse.ArgumentParser()
	parser.add_argument('--data_path', default='./Data/LDC2006T06/data/Chinese/')

	args = parser.parse_args()

	data_path = args.data_path