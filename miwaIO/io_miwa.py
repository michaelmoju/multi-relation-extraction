import sys
import re
from miwaIO.sentence import Token, Sentence
from miwaIO.annotation import *





def read_so(fh):
	out_sents = []
	with open(fh, 'r') as f:
		mySent = None
		for line in f:
			if line == '\n': continue
			else:
				data = line.rstrip().split()
			if len(data) == 5:
				if mySent:
					out_sents.append(mySent)
					del mySent
				start, end, _, id, _ = data
				mySent = Sentence(start, end, id[4:-1])
			elif len(data) == 7:
				start, end, _, id, word, _, _ = data
				myToken = Token(start, end, id[4:-1], word[6:-1])
				mySent.append_token(myToken)
			else:
				print("len(data):{}".format(len(data)))
				sys.stderr.write("read_so error!")
	return out_sents


def read_annot(fh):
	entity_mentions = {}
	relation_mentions = {}

	with open(fh, 'r') as f:
		for line in f:
			if line == '\n': continue
			data = line.rstrip().split()
			rgx_entity = re.compile('(.*)-(E[0-9]+)-([0-9]+)')
			rgx_relation = re.compile('(.*)-(R[0-9]+)-([0-9]+)')

			if rgx_entity.search(data[0]): #entity
				docID, *_ = rgx_entity.match(data[0]).groups()
				entity_mentions[data[0]] = EntityMention(data[0], data[1], data[2], data[3], data[4:])
			elif rgx_relation.search(data[0]): #relation
				relation_mentions[data[0]] = RelationMention(data[0], data[1], data[2], data[3])
			else:
				print(len(data))
				sys.stderr.write("read_annot error!")

	print("docID:{} has {} entity mentions".format(docID, len(entity_mentions)))
	print("docID:{} has {} relation mentions".format(docID, len(relation_mentions)))
	print()
	return entity_mentions, relation_mentions


def check_no_nested_entity_mentions(entity_mentions):
	def nested(m, start, end):
		for i in range(start, end + 1):
			if i in range(m.start, m.end + 1):
				return True

	span_list = []

	for m in entity_mentions.values():
		for (start, end) in span_list:
			if not nested(m, start, end): continue
			else: sys.stderr.write("nested entities found!")
		span_list.append((m.start, m.end))


if __name__ == '__main__':
	mysents = read_so("/media/moju/data/work/resource/data/ace-2005/miwa2016/corpus/dev/AFP_ENG_20030327.0224.split.stanford.so")
	print(len(mysents))

	entity_mentions, relation_mentions = read_annot("/media/moju/data/work/resource/data/ace-2005/miwa2016/corpus/dev/AFP_ENG_20030327.0224.split.ann")

	check_no_nested_entity_mentions(entity_mentions)






