class EntityMention:
	def __init__(self, id, type, start, end, words):
		self.id = id
		self.type = type
		self.start = int(start)
		self.end = int(end)
		self.words = words

	def is_nested(self, m):
		for i in range(self.start, self.end + 1):
			if i in range(m.start, m.end + 1):
				return True
		return False


class RelationMention:
	def __init__(self, id, type, arg1, arg2):
		self.id = id
		self.type = type

		split = lambda arg: arg.split(":")
		self.arg1 = split(arg1)[1]
		self.arg2 = split(arg2)[1]

