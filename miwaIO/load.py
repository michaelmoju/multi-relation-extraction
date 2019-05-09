import operator
import os
from miwaIO.io_miwa import read_so, read_annot, check_no_nested_entity_mentions
from miwaIO.instance import *
from tensorflow.python.keras import utils


def load_entity_instance(mentions, sent):
	mention_in_sent = []
	for m in mentions:
		if sent.start <= m.start <= sent.end:
			assert sent.start <= m.end <= sent.end
			mention_in_sent.append(m)
	return EntityInstance(sent, mention_in_sent)


def load_relation_instance(r_mention, e_mentions, sents):
	arg1 = e_mentions[r_mention.arg1]
	arg2 = e_mentions[r_mention.arg2]

	if arg1.is_nested(arg2):
		print(arg1.start)
		print(arg1.words)
		print(arg2.start)
		print(arg2.words)
		print("r_mention_id:{}".format(r_mention.id))

	else:
		for sent in sents:
			if sent.start <= arg1.start <= sent.end:
				assert sent.start <= arg1.end <= sent.end
				assert sent.start <= arg2.start <= sent.end
				assert sent.start <= arg2.end <= sent.end

				return RelationInstance(sent, r_mention.type, arg1, arg2)
			

def load_sentence_relation_instance(sent, r_mentions, e_mentions):
	entity_lst = []
	for e in e_mentions.values():
		if sent.start <= e.start <= sent.end:
			assert sent.start <= e.end <= sent.end
			assert sent.start <= e.start <= sent.end
			assert sent.start <= e.end <= sent.end
			entity_lst.append(e)
	
	relation_lst = []
	for r in r_mentions.values():
		arg1 = e_mentions[r.arg1]
		arg2 = e_mentions[r.arg2]
		
		if arg1.is_nested(arg2):
			continue
			# print(arg1.start)
			# print(arg1.words)
			# print(arg2.start)
			# print(arg2.words)
			# print("r_mention_id:{}".format(r.id))
		
		else:
			if sent.start <= arg1.start <= sent.end:
				assert sent.start <= arg1.end <= sent.end
				assert sent.start <= arg2.start <= sent.end
				assert sent.start <= arg2.end <= sent.end
				relation_lst.append(r)
	entity_lst.sort(key=operator.attrgetter('start'))
	
	return SentenceRelationInstance(sent, entity_lst, relation_lst, e_mentions)


def load_entity_instances_from_file(dir, docID):
	out_instances = []
	entity_num = 0
	relation_num = 0
	sentence_num = 0

	entity_mentions, relation_mentions = read_annot(dir+docID+'.split.ann')
	check_no_nested_entity_mentions(entity_mentions, docID)
	entity_num += len(entity_mentions)
	relation_num += len(relation_mentions)

	mysents = read_so(dir+docID +'.split.stanford.so')
	sentence_num += len(mysents)

	for sent in mysents:
		out_instances.append(load_entity_instance(entity_mentions.values(), sent))

	print("entity number:{}".format(entity_num))
	print("relation number:{}".format(relation_num))
	print("sentence number:{}".format(sentence_num))

	return out_instances


def load_relation_instances_from_file(dir, docID):
	out_instances = []
	entity_mentions, relation_mentions = read_annot(dir + docID + '.split.ann')
	check_no_nested_entity_mentions(entity_mentions, docID)
	mysents = read_so(dir+docID +'.split.stanford.so')

	for r_mention in relation_mentions.values():
		r_instance = load_relation_instance(r_mention, entity_mentions, mysents)
		if r_instance:
			out_instances.append(r_instance)

	return out_instances


def load_sentence_relation_instances_from_file(dir, docID):
	out_instances = []
	entity_mentions, relation_mentions = read_annot(dir + docID + '.split.ann')
	check_no_nested_entity_mentions(entity_mentions, docID)
	mysents = read_so(dir + docID + '.split.stanford.so')
	
	for sent in mysents:
		r_instance = load_sentence_relation_instance(sent, relation_mentions, entity_mentions)
		out_instances.append(r_instance)
	
	return out_instances


def load_relation_ext_instances_from_file(dir, docID):
	out_instances = []
	entity_mentions, relation_mentions = read_annot(dir + docID + '.split.ann')
	check_no_nested_entity_mentions(entity_mentions, docID)
	mysents = read_so(dir + docID + '.split.stanford.so')

	for sent in mysents:
		r_instance = load_sentence_relation_instance(sent, relation_mentions, entity_mentions)
		out_instances += r_instance.relation_ext_lst
	
	return out_instances


def load_entity_instances_from_files(dir, max_sent=False):
	out_instances = []
	entity_num = 0
	relation_num = 0
	sentence_num = 0
	max_sent_token_num = 0

	# TODO: load it from docIDs
	for f in os.listdir(dir):
		if f.endswith('.split.ann'):
			entity_mentions, relation_mentions = read_annot(dir+f)
			check_no_nested_entity_mentions(entity_mentions, docID=f)
			entity_num += len(entity_mentions)
			relation_num += len(relation_mentions)

			mysents = read_so(dir+f[:-10]+'.split.stanford.so')
			sentence_num += len(mysents)

			for sent in mysents:
				if max_sent:
					if len(sent.tokens) > max_sent_token_num:
						max_sent_token_num = len(sent.tokens)

				out_instances.append(load_entity_instance(entity_mentions.values(), sent))


	print("entity number:{}".format(entity_num))
	print("relation number:{}".format(relation_num))
	print("sentence number:{}".format(sentence_num))
			# print(out_sents[0].id)

	if max_sent: return out_instances, max_sent
	else: return out_instances


def load_relation_instances_from_files(dir):
	out_instances = []
	for f in os.listdir(dir):
		if f.endswith('.split.ann'):
			out_instances += load_relation_instances_from_file(dir, f[:-10])

	return out_instances


def load_sentence_relation_instances_from_files(dir):
	out_instances = []
	for f in os.listdir(dir):
		if f.endswith('.split.ann'):
			out_instances += load_sentence_relation_instances_from_file(dir, f[:-10])
	
	return out_instances


def load_relation_ext_instances_from_files(dir):
	out_instances = []
	for f in os.listdir(dir):
		if f.endswith('.split.ann'):
			out_instances += load_relation_ext_instances_from_file(dir, f[:-10])

	return out_instances


def load_data_from_path(data_path, word2idx, load_instance, to_indices, type_n):
	# get entity/relation instances
	train_instances = load_instance(data_path + 'train/')
	dev_instances = load_instance(data_path + 'dev/')
	test_instances = load_instance(data_path + 'test/')
	# get x and label
	*train_x, train_label = to_indices(train_instances, word2idx)
	*dev_x, dev_label = to_indices(dev_instances, word2idx)
	*test_x, test_label = to_indices(test_instances, word2idx)
	# label to one-hot
	train_y = utils.to_categorical(train_label, type_n)
	dev_y = utils.to_categorical(dev_label, type_n)
	test_y = utils.to_categorical(test_label, type_n)

	return train_x, train_y, dev_x, dev_y, test_x, test_y

if __name__ == '__main__':
	# out_instances = load_entity_instances_from_file('../../resource/data/ace-2005/miwa2016/corpus/dev/', 'XIN_ENG_20030513.0002')
	# print(out_instances[5].get_label(label2idx, 200))
	# print(out_instances[5].get_tokens())

	# out_instances = load_entity_instances_from_files('../../resource/data/ace-2005/miwa2016/corpus/dev/')
	# print(out_instances[10].get_label(label2idx, 200))
	# print(out_instances[10].get_tokens())
#======================================================================
	# max_sent_num = 0
	#
	# out_instances, max_sent_num = load_entity_instances_from_files('../../resource/data/ace-2005/miwa2016/corpus/train/', max_sent_num)
	# print(max_sent_num)
	#
	# max_sent_num = 0
	# out_instances, max_sent_num = load_entity_instances_from_files('../../resource/data/ace-2005/miwa2016/corpus/dev/', max_sent_num)
	# print(max_sent_num)
	#
	# max_sent_num = 0
	# out_instances, max_sent_num = load_entity_instances_from_files('../../resource/data/ace-2005/miwa2016/corpus/test/', max_sent_num)
	# print(max_sent_num)
#======================================================================

	# out_instances= load_relation_instances_from_files(
	# 	'../../resource/data/ace-2005/miwa2016/corpus/dev/')
	#
	# print(len(out_instances))
	#
	# out_instances = load_relation_instances_from_file(
	# 	'../../resource/data/ace-2005/miwa2016/corpus/dev/', 'XIN_ENG_20030513.0002')
	#
	# print(len(out_instances))
	#
	# print(out_instances[0].sentence.to_words())
	# print(out_instances[0].type)
	# print(out_instances[0].arg1.words)
	# print(out_instances[0].arg2.words)

#=======================================================================
	out_instances= load_relation_ext_instances_from_files(
		'../../resource/data/ace-2005/miwa2016/corpus/dev/')

	print(len(out_instances))

	out_instances = load_sentence_relation_instances_from_file(
		'../../resource/data/ace-2005/miwa2016/corpus/dev/', 'AFP_ENG_20030327.0224')

	print(len(out_instances))
	#
	# for instance in out_instances:
	# 	instance.dump()

