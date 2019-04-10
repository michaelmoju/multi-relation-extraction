import numpy as np
import os
import json

module_location = os.path.abspath(__file__)
module_location = os.path.dirname(module_location)

with open(os.path.join(module_location, "../model_params.json")) as f:
	model_params = json.load(f)

# with open(os.path.join(module_location, "../structure/", model_params["property2idx"])) as f:
# 	property2idx = ast.literal_eval(f.read())


def get_entity_type_from_id(entityid, entity_dir):
	print(entityid)

	# TODO: revise this with re
	*docID_ls, eID, mID = entityid.split('-')

	docID = ''
	for i in docID_ls:
		docID += i + '-'

	docID = docID[:-1]

	print(docID)

	with open(entity_dir + docID + '.apf.xml.entity.json') as f:
		entity_dic = json.load(f)
		entity_type = entity_dic[docID + '-' + eID]['entityType']

	return entity_type


def get_entity_idx(r, entity_dir):
	label2idx = {'B-PER': 1, 'I-PER': 2, 'L-PER': 3, 'U-PER': 4,
				 'B-ORG': 5, 'I-ORG': 6, 'L-ORG': 7, 'U-ORG': 8,
				 'B-LOC': 9, 'I-LOC': 10, 'L-LOC': 11, 'U-LOC': 12,
				 'B-GPE': 13, 'I-GPE': 14, 'L-GPE': 15, 'U-GPE': 16,
				 'B-FAC': 17, 'I-FAC': 18, 'L-FAC': 19, 'U-FAC': 20,
				 'B-VEH': 21, 'I-VEH': 22, 'L-VEH': 23, 'U-VEH': 24,
				 'B-WEA': 25, 'I-WEA': 26, 'L-WEA': 27, 'U-WEA': 28, 'O': 29}

	mention1_idx = np.zeros((model_params['max_sent_len'],))
	mention2_idx = np.zeros((model_params['max_sent_len'],))

	for i in range(len(r['Tokens'])):
		mention1_idx[i] = label2idx['O']
		mention2_idx[i] = label2idx['O']
	e = r['mentionArg1']
	for i in range(e['start'], e['end'] + 1):
		if e['end'] == e['start']:
			key = 'U'
		elif i == e['start']:
			key = 'B'
		elif i == e['end']:
			key = 'L'
		else:
			key = 'I'
		key = key + '-' + get_entity_type_from_id(e['argMentionid'], entity_dir)
		mention1_idx[i] = label2idx[key]

	e2 = r['mentionArg2']
	for i in range(e2['start'], e2['end'] + 1):
		if e2['end'] == e2['start']:
			key = 'U'
		elif i == e2['start']:
			key = 'B'
		elif i == e2['end']:
			key = 'L'
		else:
			key = 'I'
		key = key + '-' + get_entity_type_from_id(e2['argMentionid'], entity_dir)
		mention2_idx[i] = label2idx[key]

	return mention1_idx, mention2_idx


def load_entity_instances(entity_dir):
	entity_dicts = io.get_entity_from_files(entity_dir)
	entities = []
	entity_instances = {}
	nested_sentence_set = set()
	for entity_id, entity_dict in entity_dicts.items():
		entities.append(Entity().set_from_entity_dict(entity_dict))

	print(entities)
	for entity in entities:
		rgx = re.compile('(.*)-(E[0-9]+)')
		docID, _ = rgx.match(entity.id).groups()
		for mention in entity.mentions:
			sentenceID = docID + '-' + str(mention.sentence_index)
			if sentenceID not in entity_instances.keys():
				entity_instances[sentenceID] = EntityInstance(sentenceID, len(mention.tokens))
			else:
				entity_instances[sentenceID].add_mention(mention)
				if entity_instances[sentenceID].is_nested:
					print('nested entity mention found in {}'.format(sentenceID))
					nested_sentence_set.add(sentenceID)
	return entity_instances, nested_sentence_set

