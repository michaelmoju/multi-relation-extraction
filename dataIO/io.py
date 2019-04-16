# Author: Moju Wu

import json
import glob


def read_relations_from_file(json_file):
	data = []
	with open(json_file) as f:
		data += json.load(f)
	return data


def load_relation_from_existing_sets(relationMention_ph):

	train_data = read_relations_from_file(relationMention_ph+'train.relationMention.json')
	val_data = read_relations_from_file(relationMention_ph+'val.relationMention.json')
	test_data = read_relations_from_file(relationMention_ph+'test.relationMention.json')

	print("Train set size:", len(train_data))
	print("Val set size:", len(val_data))
	print("Test set size:", len(test_data))
	return train_data, val_data, test_data


def get_entity_from_file(fp, entity_id):
	entity_dic = json.load(fp)
	return entity_dic[entity_id]


def get_entity_from_files(path):
	entities = {}
	files = glob.glob(path+'*.entity.json')
	for file in files:
		with open(file) as f:
			entities.update(json.load(f))
	print(entities.keys())

	return entities


