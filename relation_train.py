import my_models
import word_embeddings
import config
import random
from dataIO import *

random.seed(0)

if __name__ == '__main__':
	import argparse

	parser = argparse.ArgumentParser()
	parser.add_argument('model_name')
	parser.add_argument('mode', choices=['train-entity', 'train-relation', 'evaluate'])
	parser.add_argument('--data_path', default='./Data/LDC2006T06/data/Chinese/')
	parser.add_argument('--embedding', default='../resource/embeddings/glove/glove.6B.50d.txt')
	args = parser.parse_args()

	embeddings, word2idx = word_embeddings.load_word_emb(args.embedding)

	model_name = args.model_name
	mode = args.mode
	data_path = args.data_path

	config = config.define_config()

	if mode == 'train-entity':
		model = getattr(my_models, model_name)(embeddings,  lstm_size = 128, entity_type_n=29, max_sent_len=config.max_sent_len, dropout=False)



