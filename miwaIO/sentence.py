class Sentence:
	def __init__(self, start, end, id, docID):
		self.docID = docID
		self.start = int(start)
		self.end = int(end)
		self.id = id
		self.tokens = []

	def append_token(self, token):
		self.tokens.append(token)

	def to_words(self):
		return [t.word for t in self.tokens]
	
	def __str__(self):
		return 'id:{}\t'.format(self.id) + 'start:{}\t'.format(self.start) + 'end:{}\t'.format(self.end) +\
		       'sentence:{}\n'.format(' '.join([t.word for t in self.tokens]))


class Token:
	def __init__(self, start, end, id, word):
		self.start = int(start)
		self.end = int(end)
		self.id = id
		self.word = word




