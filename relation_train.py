import my_models
import word_embeddings
import config
from miwaIO.instances_get import *
import my_models
import tensorflow as tf
import tensorflow.keras as keras

if __name__ == '__main__':
	import argparse

	parser = argparse.ArgumentParser()
	parser.add_argument('model_name')
	parser.add_argument('mode', choices=['train-entity', 'train-relation', 'evaluate'])
	parser.add_argument('--data_path', default='../resource/data/ace-2005/miwa2016/corpus/')
	parser.add_argument('--embedding', default='../resource/embeddings/glove/glove.6B.50d.txt')
	args = parser.parse_args()

	embeddings, word2idx = word_embeddings.load_word_emb(args.embedding)

	model_name = args.model_name
	mode = args.mode
	data_path = args.data_path

	train_dir = data_path + 'train/'
	dev_dir = data_path + 'dev/'
	test_dir = data_path + 'test/'

	train_instances = load_entity_instances_from_files(train_dir)
	dev_instances = load_entity_instances_from_files(dev_dir)
	test_instances = load_entity_instances_from_files(test_dir)

	sent_matrix, label_matrix = my_models.to_indices_with_entity_instances(train_instances, word2idx)

	print(tf.one_hot([0, 29], 29).numpy())

	exit(0)


	if mode == 'train-entity':
		model = getattr(my_models, model_name)(embeddings,  lstm_size = 128, entity_type_n=29, max_sent_len=config.max_sent_len, dropout=False)



