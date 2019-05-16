import word_embeddings
from miwaIO.load import *
import my_models
from my_models import r_to_indices_position_e
from tensorflow.python.keras import models, callbacks, optimizers
import matplotlib.pyplot as plt
from miwaIO.instance import r_idx2label, RelationExtInstance

p = my_models.p


def prediction(model, data_dir1_input, data_dir2_input):
	dir1_prediction = model.predict(data_dir1_input, batch_size=1, verbose=1)
	dir2_prediction = model.predict(data_dir2_input, batch_size=1, verbose=1)
	
	dir1_max = np.max(dir1_prediction, axis=-1)
	dir2_max = np.max(dir2_prediction, axis=-1)
	
	if dir1_max >= dir2_max:
		predict_logit = np.argmax(dir1_prediction, axis=-1)
		
	else:
		predict_logit = np.argmax(dir2_prediction, axis=-1)
		
	return r_idx2label[predict_logit]


def evaluate_micro_f(model, sentence_instances, gold_outputs, word2idx):
	tp = 0
	tn = 0
	fn = 0
	fp = 0
	for sentence_instance, label in sentence_instances, gold_outputs:
		for arg1_idx in range(len(sentence_instance.entity_lst) - 1):
			for arg2_idx in range(arg1_idx + 1, len(sentence_instance.entity_lst)):
				e_tuple = (sentence_instance.entity_lst[arg1_idx], sentence_instance.entity_lst[arg2_idx])
				dir1_instance = RelationExtInstance(label, e_tuple[0], e_tuple[1], sentence_instance.sentence)
				dir2_instance = RelationExtInstance(label, e_tuple[0], e_tuple[1], sentence_instance.sentence)
				dir1_instance_indices = r_to_indices_position_e([dir1_instance], word2idx)
				dir2_instance_indices = r_to_indices_position_e([dir2_instance], word2idx)
				
				y_hat = prediction(model, dir1_instance_indices, dir2_instance_indices)
				if y_hat == 'NONE':
					if label == 'NONE':
						tn += 1
					else:
						fn += 1
				else:
					if y_hat == label:
						tp += 1
					else:
						fp += 1
	precision = tp/(tp+fp)
	recall = tp/(tp+fn)
	return 2.0 * precision * recall / (precision + recall)
	
	
def plot_callback(callback_history, models_folder):

	# Plot training & validation accuracy values
	plt.plot(callback_history.history['acc'])
	plt.plot(callback_history.history['val_acc'])
	plt.title('Model accuracy')
	plt.ylabel('Accuracy')
	plt.xlabel('Epoch')
	plt.legend(['Train', 'Validation'], loc='upper left')
	plt.savefig(args.models_folder + 'accuracy.png')
	plt.clf()
	# plt.show()

	# Plot training & validation loss values
	plt.plot(callback_history.history['loss'])
	plt.plot(callback_history.history['val_loss'])
	plt.title('Model loss')
	plt.ylabel('Loss')
	plt.xlabel('Epoch')
	plt.legend(['Train', 'Validation'], loc='upper left')
	plt.savefig(models_folder + 'loss.png')
	# plt.show()
	plt.clf()
	
def load_eType_embeddings():
	embeddings = np.zeros((9, 8), dtype='float32')
	
	for i in range(1, 9):
		embeddings[i, i - 1] = 1
	return embeddings


if __name__ == '__main__':
	import argparse

	parser = argparse.ArgumentParser()
	parser.add_argument('model_name')
	parser.add_argument('mode', choices=['train-entity', 'train-relation', 'evaluate', 'predict'])
	parser.add_argument('--epoch', default=1, type=int)
	parser.add_argument('--data_path', default='../resource/data/ace-2005/miwa2016/corpus/')
	# parser.add_argument('--embedding', default='../resource/embeddings/glove/glove.6B.50d.txt')
	parser.add_argument('--embedding', default='../resource/embeddings/tticoin/wikipedia200.txt')
	parser.add_argument('--metadata', default='01', type=str)
	parser.add_argument('--checkpoint', action='store_true')
	parser.add_argument('--dropout', action='store_true')
	parser.add_argument('--models_folder', default="./trainedmodels/entity4/")
	parser.add_argument('--entity_folder', default='./trainedmodels/entity4/')
	parser.add_argument('--train_entity', action='store_true')
	parser.add_argument('--learning_rate', default='0.001', type=float)
	args = parser.parse_args()
	
	if 'tticoin' in args.embeddings:
		embeddings, word2idx = word_embeddings.load_word_emb_miwa(args.embedding)
	else:
		embeddings, word2idx = word_embeddings.load_word_emb(args.embedding)
	
	model_name = args.model_name
	mode = args.mode
	data_path = args.data_path

	cbfunctions = []
	if args.checkpoint:
		checkpoint = callbacks.ModelCheckpoint(args.models_folder + model_name + '-{epoch:02d}' + ".kerasmodel",
											   monitor='val_loss', verbose=1, save_best_only=True)
		cbfunctions.append(checkpoint)

	if mode == 'train-entity':
		load_instance = load_entity_instances_from_files
		to_indices = my_models.to_indices_with_entity_instances
		model = my_models.model_entity(embeddings, dropout=args.dropout)
		data = load_data_from_path(data_path, word2idx, load_instance, to_indices, p['entity_type_n'])

	elif mode == 'train-relation':
		# load_instance = load_relation_instances_from_files
		load_instance = load_relation_ext_instances_from_files

		if 'LSTMbaseline' in model_name:
			to_indices = my_models.r_to_indices_position_e
			model = my_models.model_relation_LSTMbaseline(embeddings)
		
		elif 'LSTMtype' in model_name:
			to_indices = my_models.r_to_indices_typed_e
			eType_embeddings = load_eType_embeddings()
			model = my_models.model_relation_LSTMtype(embeddings, eType_embeddings)

		elif 'multi' in model_name:
			entity_model = models.load_model(args.entity_folder + "model_entity" + "-" + args.metadata + ".kerasmodel")
			entity_weights = entity_model.get_layer(name='entity_BiLSTM_layer').get_weights()

			to_indices = my_models.r_to_indices_type_e
			model = my_models.model_relation_multi(embeddings, entity_weights)
		elif model_name == 'model_relation_entity_LSTM':
			entity_model = models.load_model(
				args.entity_folder + "model_entity" + "-" + args.metadata + ".kerasmodel")
			entity_weights = entity_model.get_layer(name='entity_BiLSTM_layer').get_weights()

			if args.train_entity:
				to_indices = my_models.r_to_indices_e_mat_train_entity
				model = my_models.model_relation_entity_LSTM(embeddings, entity_weights, train_entity=True, dropout=False)
			else:
				to_indices = my_models.r_to_indices_e_mat
				model = my_models.model_relation_entity_LSTM(embeddings, entity_weights, train_entity=False, dropout=False)

		else:
			raise NameError
		
		data = load_data_from_path(data_path, word2idx, load_instance, to_indices, p['relation_type_n'])
		print("train:{}".format(len(data[1])))
		print("dev:{}".format(len(data[3])))
		print("test:{}".format(len(data[5])))
	else:
		raise NameError
	
	if 'train' in mode:
		adamopt = optimizers.Adam(args.learning_rate)
		model.compile(optimizer=adamopt, loss='categorical_crossentropy', metrics=['accuracy'])

		callback_history = model.fit(data[0], data[1],
											  epochs=args.epoch,
											  batch_size=p['batch_size'],
											  verbose=1,
											  validation_data=(data[2], data[3]),
											  callbacks=cbfunctions)
	
		plot_callback(callback_history, args.models_folder)
	
	elif mode == "eval":
		test_dir = data_path + 'test/'
		
		sentence_instances = load_sentence_relation_instances_from_files(test_dir)
		
		print("Loading the best model")
		model = getattr(my_models, model_name)(embeddings)
		model.load_weights(args.models_folder + model_name + "-" + args.metadata + ".kerasmodel")
		
		micro_f1 = evaluate_micro_f(model, sentence_instances, gold_outputs, word2idx)

	elif mode == 'summary':
		model = getattr(my_models, model_name)(embeddings)
		print(model.summary())

	elif mode == "predict":

		print("Loading the best model")
		model = models.load_model(args.models_folder + model_name + "-" + args.metadata + ".kerasmodel")
		print(dev_instances[10].get_tokens())
		print(dev_instances[10].label)

		predictions = prediction(model, dev_sent)

		print(predictions[10])