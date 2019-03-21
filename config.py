
class AttrDict(dict):
	__getattr__ = dict.__getitem__
	__setattr__ = dict.__setitem__


def define_config():
	config = AttrDict()
	config.max_sent_len = 200
	config.word_emb_dim = 50
	config.entity_emb_dim = 3
	return config



