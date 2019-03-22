import models
import random
import dataIO.io

random.seed(0)

if __name__ == '__main__':
	import argparse

	parser = argparse.ArgumentParser()
	parser.add_argument('model_name')
	parser.add_argument('mode', choices=['train-entity', 'train-relation', 'evaluate'])
	parser.add_argument('--data_path', default='./Data/LDC2006T06/data/Chinese/')
	args = parser.parse_args()

	model_name = args.model_name
	mode = args.mode
	data_path = args.data_path

	if mode == 'train-entity':
		models.run_entity_model()



