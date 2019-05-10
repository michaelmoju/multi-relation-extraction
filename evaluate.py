import word_embeddings
from miwaIO.load import *
import my_models
import tensorflow as tf
from tensorflow.python.keras import models, callbacks, utils, optimizers
import matplotlib.pyplot as plt

p = my_models.p


def compute_micro_PRF(predicted_idx, gold_idx, i=-1, empty_label=None):
	if i == -1:
		i = len(predicted_idx)
	if i < len(gold_idx):
		predicted_idx = np.concatenate([predicted_idx[:i], np.ones(len(gold_idx) - i)])
	t = predicted_idx != empty_label
	tp = len((predicted_idx[t] == gold_idx[t]).nonzero()[0])
	tp_fp = len((predicted_idx != empty_label).nonzero()[0])
	tp_fn = len((gold_idx != empty_label).nonzero()[0])
	prec = (tp / tp_fp) if tp_fp != 0 else 1.0
	rec = tp / tp_fn if tp_fp != 0 else 0.0
	f1 = 0.0
	if (rec + prec) > 0:
		f1 = 2.0 * prec * rec / (prec + rec)
	return prec, rec, f1


def compute_macro_PRF(predicted_idx, gold_idx, i=-1, empty_label=None):
	if i == -1:
		i = len(predicted_idx)
	
	complete_rel_set = set(gold_idx) - {0, empty_label}
	avg_prec = 0.0
	avg_rec = 0.0
	
	for r in complete_rel_set:
		r_indices = (predicted_idx[:i] == r)
		tp = len((predicted_idx[:i][r_indices] == gold_idx[:i][r_indices]).nonzero()[0])
		tp_fp = len(r_indices.nonzero()[0])
		tp_fn = len((gold_idx == r).nonzero()[0])
		prec = (tp / tp_fp) if tp_fp > 0 else 0
		rec = tp / tp_fn
		avg_prec += prec
		avg_rec += rec
	f1 = 0
	avg_prec = avg_prec / len(set(predicted_idx[:i]))
	avg_rec = avg_rec / len(complete_rel_set)
	if (avg_rec + avg_prec) > 0:
		f1 = 2.0 * avg_prec * avg_rec / (avg_prec + avg_rec)
	
	return avg_prec, avg_rec, f1


def predict(model, data_input):
	outs = model.predict(data_input, batch_size=my_models.p['batch_size'], verbose=1)
	predictions = np.argmax(outs, axis=-1)
	
	return predictions


def evaluate_relation(model, labels, x):
	predictions = predict(model, x)
	labels = np.argmax(labels, axis=-1)

	print(set(predictions))
	print(set(labels))
	
	print("Macro F:{}".format(compute_macro_PRF(predictions, labels, empty_label=0)))
	print("Micro F:{}".format(compute_micro_PRF(predictions, labels, empty_label=0)))
	
	
def evaluate_entity(model, labels, x):
	predictions = predict(model, x)
	labels = np.argmax(labels, axis=-1)
	
	pred_labels = []
	gold_labels = []
	
	for i, prediction in enumerate(predictions):
		pred_labels.append([my_models.e_idx2label[word] for word in prediction])
		gold_labels.append([my_models.e_idx2label[word] for word in labels[i]])
	
	print(pred_labels[100])
	print(gold_labels[100])
	

if __name__ == '__main__':
	import argparse
	
	parser = argparse.ArgumentParser()
	parser.add_argument('model_name')
	parser.add_argument('mode', choices=['entity', 'relation'])
	parser.add_argument('--out', default='./trainedmodels/eval')
	parser.add_argument('--data_path', default='../resource/data/ace-2005/miwa2016/corpus/')
	parser.add_argument('--embedding', default='../resource/embeddings/glove/glove.6B.50d.txt')
	parser.add_argument('--models_folder', default="./trainedmodels/entity4/")
	parser.add_argument('--metadata', default='01', type=str)
	parser.add_argument('--train_entity', action='store_true')
	args = parser.parse_args()
	
	mode = args.mode
	model_name = args.model_name
	data_path = args.data_path
	
	embeddings, word2idx = word_embeddings.load_word_emb(args.embedding)
	
	if mode == 'entity':
		load_instance = load_entity_instances_from_files
		to_indices = my_models.to_indices_with_entity_instances
		data = load_data_from_path(data_path, word2idx, load_instance, to_indices, p['entity_type_n'])
		
		print("Loading the best entity model...")
		model = models.load_model(args.models_folder + model_name + "-" + args.metadata + ".kerasmodel")
		
		evaluate_entity(model, data[5], data[4])
	
	elif mode == 'relation':
		# load_instance = load_relation_instances_from_files
		load_instance = load_relation_ext_instances_from_files
		
		if 'LSTMbaseline' in model_name:
			to_indices = my_models.r_to_indices_position_e
		
		elif 'LSTMtype' in model_name:
			to_indices = my_models.r_to_indices_position_e
		
		elif 'multi' in model_name:
			to_indices = my_models.r_to_indices_type_e
		
		elif model_name == 'model_relation_entity_LSTM':
			if args.train_entity:
				to_indices = my_models.r_to_indices_e_mat_train_entity
			else:
				to_indices = my_models.r_to_indices_e_mat
		
		else:
			raise Exception
		
		data = load_data_from_path(data_path, word2idx, load_instance, to_indices, p['relation_type_n'])
		
		print("Loading the best relation model...")
		model = models.load_model(args.models_folder + model_name + "-" + args.metadata + ".kerasmodel")
	
		evaluate_relation(model, data[5], data[4])
	else:
		raise Exception
