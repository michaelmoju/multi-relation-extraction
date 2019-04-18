import word_embeddings
from miwaIO.load import *
import my_models
from tensorflow.keras import models, callbacks, utils
import matplotlib.pyplot as plt

p = my_models.p


def prediction(model, data_input):
	predictions = model.predict(data_input, batch_size=my_models.p['batch_size'], verbose=1)

	predictions_classes = np.argmax(predictions, axis=-1)

	return predictions_classes


def plot_callback(callback_history, models_folder):

	# Plot training & validation accuracy values
	plt.plot(callback_history.history['accuracy'])
	plt.plot(callback_history.history['val_accuracy'])
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


if __name__ == '__main__':
	import argparse

	parser = argparse.ArgumentParser()
	parser.add_argument('model_name')
	parser.add_argument('mode', choices=['train-entity', 'train-relation', 'evaluate', 'predict'])
	parser.add_argument('--epoch', default=50, type=int)
	parser.add_argument('--data_path', default='../resource/data/ace-2005/miwa2016/corpus/')
	parser.add_argument('--embedding', default='../resource/embeddings/glove/glove.6B.50d.txt')
	parser.add_argument('--metadata', default='01', type=str)
	parser.add_argument('--checkpoint', action='store_true')
	parser.add_argument('--dropout', action='store_true')
	parser.add_argument('--pretrain', action='store_true')
	parser.add_argument('--models_folder', default="./trainedmodels/entity/")
	parser.add_argument('--entity_folder', default='./trainedmodels/relation/')
	args = parser.parse_args()

	embeddings, word2idx = word_embeddings.load_word_emb(args.embedding)

	model_name = args.model_name
	mode = args.mode
	data_path = args.data_path

	train_dir = data_path + 'train/'
	dev_dir = data_path + 'dev/'
	test_dir = data_path + 'test/'

	cbfunctions = []
	if args.checkpoint:
		checkpoint = callbacks.ModelCheckpoint(args.models_folder + model_name + '-{epoch:02d}' + ".kerasmodel",
											   monitor='val_loss', verbose=1, save_best_only=True)
		cbfunctions.append(checkpoint)

	if 'train' in mode:
		if mode == 'train-entity':
			load_instance = load_entity_instances_from_files
			to_indices = my_models.to_indices_with_entity_instances
			model = my_models.model_entity(embeddings, dropout=args.dropout)
			data = load_data_from_path(data_path, word2idx, load_instance, to_indices, p['entity_type_n'])

		elif mode == 'train-relation':
			load_instance = load_relation_instances_from_files

			if 'LSTMbaseline' in model_name:
				to_indices = my_models.r_to_indices_position_e
				model = my_models.model_relation_LSTMbaseline(embeddings)

			elif 'multi' in model_name:
				entity_model = models.load_model(args.entity_folder + "model_entity" + "-" + args.metadata + ".kerasmodel")
				entity_weights = entity_model.get_layer(name='entity_BiLSTM_layer').get_weights()

				to_indices = my_models.r_to_indices_type_e
				model = my_models.model_relation_multi(embeddings, entity_weights)

			else:
				raise NameError

			data = load_data_from_path(data_path, word2idx, load_instance, to_indices, p['relation_type_n'])
		else:
			raise NameError

		callback_history = model.fit(data[0], data[1],
											  epochs=args.epoch,
											  batch_size=p['batch_size'],
											  verbose=1,
											  validation_data=(data[2], data[3]),
											  callbacks=cbfunctions)

		plot_callback(callback_history, args.models_folder)

	elif mode == 'summary':
		model = getattr(my_models, model_name)(embeddings)
		print(model.summary())

	elif mode == "predict":

		print("Loading the best model")
		model = getattr(my_models, model_name)(embeddings)
		model.load_weights(args.models_folder + model_name + "-" + args.metadata + ".kerasmodel")
		print(dev_instances[10].get_tokens())
		print(dev_instances[10].label)

		predictions = prediction(model, dev_sent)

		print(predictions[10])