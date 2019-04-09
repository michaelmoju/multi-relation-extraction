import sys
from annotation.sentence import Token, Sentence


def read_so(fh):
	with open(fh, 'w') as f:
		for line in f:
			if not line: continue
			else:
				data = line.rstrip().split(" ")
			if len(data) == 5:
				if mySent:
					yield mySent
					del mySent
				start, end, _, id, _ = data
				mySent = Sentence(start, end, id[4:-1])
			elif len(data) == 7:
				start, end, _, id, word, _, _ = data
				myToken = Token(start, end, id[4:-1], word[6:-1])
				mySent.append_token(myToken)
			else: sys.stderr.write("read_so error!")


if __name__ == '__main__':
	Sents = read_so("/media/moju/data/work/resource/data/ace-2005/miwa2016/corpus/dev/AFP_ENG_20030327.0224.ann")