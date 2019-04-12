class Sentence:
	def __init__(self, start, end, id):
		self.start = int(start)
		self.end = int(end)
		self.id = id
		self.tokens = []

	def append_token(self, token):
		self.tokens.append(token)

class Token:
	def __init__(self, start, end, id, word):
		self.start = int(start)
		self.end = int(end)
		self.id = id
		self.word = word



