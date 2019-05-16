import word_embeddings
from miwaIO.load import *
import my_models
from tensorflow.python.keras import models
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from miwaIO.instance import r_label2idx


p = my_models.p


def write_to_file(fh, instance, yhat, true_class):
	fh.write("relationID: " + instance.sentence.docID + '-' + instance.sentence.id + '\n')
	fh.write("Sentence:{}".format(instance.sentence.to_words()) + '\n')
	fh.write("mention1:{}".format(instance.e1) + '\n')
	fh.write("mention2:{}".format(instance.e2) + '\n')
	fh.write("predict: " + yhat + '\n')
	fh.write("true: " + true_class + '\n')
	fh.write('\n')
	

def error_analysis(label_classes, predict_classes, out_folder, val_data):
	tp_n = 0
	fp_n = 0
	fn_n = 0
	ftype_n = 0
	fdir_n = 0
	
	fn_PHYS_n = 0
	fn_PART_n = 0
	fn_PER_n = 0
	fn_ORG_n = 0
	fn_ART_n = 0
	fn_GEN_n = 0
	
	with open(out_folder+'fp.txt', 'w') as fp_f, open(out_folder+'tp.txt', 'w') as tp_f,\
			open(out_folder+'fn.txt', 'w') as fn_f, open(out_folder+'ftype.txt', 'w') as ftype_f, \
			open(out_folder+'fdir.txt', 'w') as fdir_f, open(out_folder+ 'fn_PHYS.txt', 'w') as fn_PHYS, \
			open(out_folder+ 'fn_PART.txt', 'w') as fn_PART, open(out_folder+'fn_PER.txt', 'w') as fn_PER, \
			open(out_folder+ 'fn_ORG.txt', 'w') as fn_ORG, open(out_folder+ 'fn_ART.txt', 'w') as fn_ART, \
			open(out_folder+ 'fn_GEN.txt', 'w') as fn_GEN:
		for index, predict in enumerate(predict_classes):
			if predict != label_classes[index]:
				if predict == 'NONE':
					fn_n += 1
					write_to_file(fn_f, val_data[index], predict, label_classes[index])
					if label_classes[index][:-3] == 'PHYS':
						fn_PHYS_n += 1
						write_to_file(fn_PHYS, val_data[index], predict, label_classes[index])
					elif label_classes[index][:-3] == 'PART-WHOLE':
						fn_PART_n += 1
						write_to_file(fn_PART, val_data[index], predict, label_classes[index])
					elif label_classes[index][:-3] == 'PER-SOC':
						fn_PER_n += 1
						write_to_file(fn_PER, val_data[index], predict, label_classes[index])
					elif label_classes[index][:-3] == 'ORG-AFF':
						fn_ORG_n += 1
						write_to_file(fn_ORG, val_data[index], predict, label_classes[index])
					elif label_classes[index][:-3] == 'ART':
						fn_ART_n += 1
						write_to_file(fn_ART, val_data[index], predict, label_classes[index])
					elif label_classes[index][:-3] == 'GEN-AFF':
						fn_GEN_n += 1
						write_to_file(fn_GEN, val_data[index], predict, label_classes[index])
					assert fn_n == fn_PHYS_n + fn_PART_n + fn_PER_n + fn_ORG_n +fn_ART_n + fn_GEN_n
				elif predict != 'NONE':
					if label_classes[index] == 'NONE':
						fp_n += 1
						write_to_file(fp_f, val_data[index], predict, label_classes[index])
					elif predict[:-3] == label_classes[index][:-3]:
						fdir_n += 1
						write_to_file(fdir_f, val_data[index], predict, label_classes[index])
					else:
						ftype_n += 1
						write_to_file(ftype_f, val_data[index], predict, label_classes[index])
			elif predict == label_classes[index]:
				tp_n += 1
				write_to_file(tp_f, val_data[index], predict, label_classes[index])
	assert tp_n+fp_n+fn_n+ftype_n+fdir_n == len(label_classes)
	total = len(label_classes)
	print('total:{}'.format(total))
	print('tp:{} fp:{} fn:{} fdir:{} ftype:{}'.format(tp_n,fp_n,fn_n,fdir_n, ftype_n))
	print('tp:{} fp:{} fn:{} fdir:{} ftype:{}'.format(tp_n/total, fp_n/total, fn_n/total, fdir_n/total, ftype_n/total))
	print('fn_PHYS_n:{} fn_PART_n:{} fn_PER_n:{} fn_ORG_n:{} fn_ART_n:{} fn_GEN_n:{}'.format(fn_PHYS_n,fn_PART_n , fn_PER_n , fn_ORG_n ,fn_ART_n , fn_GEN_n))
def plot_comfusion_matrix(label_classes, predict_classes, out_folder):
	label_types = list(r_idx2label.values())

	cm = confusion_matrix(label_classes, predict_classes, label_types)
	print(cm)
	fig = plt.figure()
	ax = fig.add_subplot(111)
	cax = ax.matshow(cm)
	for (i, j), z in np.ndenumerate(cm):
		ax.text(j, i, '{:0.0f}'.format(z), ha='center', va='center', color='white')
	fig.colorbar(cax)
	ax.set_xticklabels([''] + label_types)
	ax.set_yticklabels([''] + label_types)
	plt.xlabel('Predicted')
	plt.ylabel('True')
	
	plt.savefig(out_folder + 'confusion_matrix.png')
	plt.show()

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


def evaluate_relation(model, labels, x, out_folder, instances):
	predictions = model.predict(x, batch_size=p['batch_size'], verbose=1)
	predictions = np.argmax(predictions, axis=-1)
	true_labels = np.argmax(labels, axis=-1)
	
	print(set(predictions))
	print(set(true_labels))
	
	print("Macro F:{}".format(compute_macro_PRF(predictions, true_labels, empty_label=0)))
	print("Micro F:{}".format(compute_micro_PRF(predictions, true_labels, empty_label=0)))
	
	label_classes = [r_idx2label.get(i) for i in true_labels]
	predict_classes = [r_idx2label.get(i) for i in predictions]
	
	plot_comfusion_matrix(label_classes, predict_classes, out_folder)
	error_analysis(label_classes, predict_classes, out_folder, instances)
	
	
def evaluate_entity(model, labels, x):
	predictions = model.predict(model, x)
	labels = np.argmax(labels, axis=-1)
	
	pred_labels = []
	gold_labels = []
	
	for i, prediction in enumerate(predictions):
		pred_labels.append([my_models.e_idx2label[word] for word in prediction])
		gold_labels.append([my_models.e_idx2label[word] for word in labels[i]])
	
	print(pred_labels[100])
	print(gold_labels[100])
	

def load_eType_embeddings():
	embeddings = np.zeros((9,8), dtype='float32')
	
	for i in range(1,9):
		embeddings[i,i-1] = 1
	return embeddings
	
	
if __name__ == '__main__':
	import argparse
	
	parser = argparse.ArgumentParser()
	parser.add_argument('model_name')
	parser.add_argument('mode', choices=['entity', 'relation'])
	parser.add_argument('--out', default='./trainedmodels/relation6/eval/')
	parser.add_argument('--data_path', default='../resource/data/ace-2005/miwa2016/corpus/')
	# parser.add_argument('--embedding', default='../resource/embeddings/glove/glove.6B.50d.txt')
	parser.add_argument('--embedding', default='../resource/embeddings/tticoin/wikipedia200.txt')
	parser.add_argument('--models_folder', default="./trainedmodels/relation6/")
	parser.add_argument('--metadata', default='33', type=str)
	parser.add_argument('--train_entity', action='store_true')
	args = parser.parse_args()
	
	mode = args.mode
	model_name = args.model_name
	data_path = args.data_path
	if 'tticoin' in args.embedding:
		embeddings, word2idx = word_embeddings.load_word_emb_miwa(args.embedding)
	else:
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
			eType_embeddings = load_eType_embeddings()
			to_indices = my_models.r_to_indices_typed_e
		
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
		
		dev_instances = load_instance(data_path + 'dev/')
		evaluate_relation(model, data[3], data[2], args.out, dev_instances)
	else:
		raise Exception
