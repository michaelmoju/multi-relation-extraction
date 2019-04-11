import numpy as np
import os
import json
from miwaIO.io_miwa import read_so, read_annot, check_no_nested_entity_mentions
from miwaIO.annotation import *

module_location = os.path.abspath(__file__)
module_location = os.path.dirname(module_location)

with open(os.path.join(module_location, "../model_params.json")) as f:
	model_params = json.load(f)


def load_entity_instance(mentions, sent):
	mention_in_sent = []
	for m in mentions:
		if sent.start <= m.start <= sent.end:
			assert sent.start <= m.end <= sent.end
			mention_in_sent.append(m)
	return EntityInstance(sent, mention_in_sent)


def load_entity_instances_from_file(dir, docID):
	out_instances = []
	entity_num = 0
	relation_num = 0
	sentence_num = 0


	entity_mentions, relation_mentions = read_annot(dir+docID+'.split.ann')
	check_no_nested_entity_mentions(entity_mentions)
	entity_num += len(entity_mentions)
	relation_num += len(relation_mentions)

	mysents = read_so(dir+docID +'.split.stanford.so')
	sentence_num += len(mysents)

	for sent in mysents:
		out_instances.append(load_entity_instance(entity_mentions.values(), sent))

	print("entity number:{}".format(entity_num))
	print("relation number:{}".format(relation_num))
	print("sentence number:{}".format(sentence_num))
	# print(out_sents[0].id)

	return out_instances

def load_entity_instances_from_files(dir):
	out_instances = []
	entity_num = 0
	relation_num = 0
	sentence_num = 0

	for f in os.listdir(dir):
		if f.endswith('.split.ann'):
			entity_mentions, relation_mentions = read_annot(dir+f)
			check_no_nested_entity_mentions(entity_mentions)
			entity_num += len(entity_mentions)
			relation_num += len(relation_mentions)

			mysents = read_so(dir+f[:-10]+'.split.stanford.so')
			sentence_num += len(mysents)

			for sent in mysents:
				out_instances.append(load_entity_instance(entity_mentions.values(), sent))


	print("entity number:{}".format(entity_num))
	print("relation number:{}".format(relation_num))
	print("sentence number:{}".format(sentence_num))
			# print(out_sents[0].id)

	return out_instances


if __name__ == '__main__':
	label2idx = {'B-PER': 1, 'I-PER': 2, 'L-PER': 3, 'U-PER': 4,
				 'B-ORG': 5, 'I-ORG': 6, 'L-ORG': 7, 'U-ORG': 8,
				 'B-LOC': 9, 'I-LOC': 10, 'L-LOC': 11, 'U-LOC': 12,
				 'B-GPE': 13, 'I-GPE': 14, 'L-GPE': 15, 'U-GPE': 16,
				 'B-FAC': 17, 'I-FAC': 18, 'L-FAC': 19, 'U-FAC': 20,
				 'B-VEH': 21, 'I-VEH': 22, 'L-VEH': 23, 'U-VEH': 24,
				 'B-WEA': 25, 'I-WEA': 26, 'L-WEA': 27, 'U-WEA': 28, 'O': 29}
	out_instances = load_entity_instances_from_file('../../resource/data/ace-2005/miwa2016/corpus/dev/', 'XIN_ENG_20030513.0002')
	print(out_instances[5].get_label(label2idx, 200))
	print(out_instances[5].get_tokens())
	# out_instances = load_entity_instances_from_files('../../resource/data/ace-2005/miwa2016/corpus/dev/')
	# print(out_instances[10].get_label(label2idx, 200))
