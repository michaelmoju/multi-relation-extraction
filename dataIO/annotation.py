class Annotation:
	def __init__(self):
		self.id = ''
		self.type = ''
		self.subtype = ''
		self.mentions =[]


class AbstractMention:
	def __init__(self):
		self.id = ''
		self.extent = ''
		self.sentence = ''
		self.tokens = []
		self.sentence_index = int
		self.start = int
		self.end = int


class EntityMention(AbstractMention):
	def __init__(self, mention):
		super().__init__()
		self.id = mention['mention_id']
		self.sentence = mention['Sentence']
		self.tokens = mention['tokens']
		self.sentence_index = mention['sentence_index']
		self.start = mention['start']
		self.end = mention['end']


class Entity(Annotation):
	def __init__(self):
		super().__init__()

	def set_from_entity_dict(self, entity_dict):
		self.id = entity_dict['entityID']
		self.type = entity_dict['entityType']
		self.subtype = entity_dict['entitySubType']

		for mention in entity_dict['entityMentionList']:
			mention = EntityMention(mention)
			self.mentions.append(mention)
		return self


class EntityInstance:
	def __init__(self, sentenceID, sentence_len):
		self.id = sentenceID
		self.sentence = ''
		self.tokens = ''
		self.exist_list = [0] * sentence_len
		self.mention_list = []
		self.is_nested = False

	def add_mention(self, mention):
		for i in range(mention.start, mention.end+1):
			print(mention.id)
			if self.exist_list[i] == 1:
				self.mention_list.append(mention)
				self.is_nested = True
			else:
				self.exist_list[i] = 1
				self.mention_list.append(mention)


class RelationInstance:
	"""
	TODO: relation instance class.
	"""

if __name__ == '__main__':
	import annotation.event

