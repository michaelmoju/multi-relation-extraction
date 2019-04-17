import word_embeddings
from miwaIO.instances_get import *
import my_models
import tensorflow as tf
from tensorflow.keras import models, callbacks
import matplotlib.pyplot as plt
from dataIO import io

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

	train_instances = load_entity_instances_from_files(train_dir)
	dev_instances = load_entity_instances_from_files(dev_dir)
	test_instances = load_entity_instances_from_files(test_dir)

	train_sent, train_label = my_models.to_indices_with_entity_instances(train_instances, word2idx)
	dev_sent, dev_label = my_models.to_indices_with_entity_instances(dev_instances, word2idx)
	# print(train_sent.shape)
	# print(train_label.shape)

	print(dev_sent.shape)

	train_one_hot = tf.keras.utils.to_categorical(train_label, 29)
	dev_one_hot = tf.keras.utils.to_categorical(dev_label, 29)

	cbfunctions = []
	if args.checkpoint:
		checkpoint = callbacks.ModelCheckpoint(args.models_folder + model_name + '-{epoch:02d}' + ".kerasmodel",
											   monitor='val_loss', verbose=1, save_best_only=True)
		cbfunctions.append(checkpoint)

	if mode == 'train-entity':
		model = getattr(my_models, model_name)(embeddings, dropout=args.dropout)
		callback_history = model.fit(train_sent, train_one_hot, epochs=args.epoch, batch_size=model_params["batch_size"],
									 validation_data=(dev_sent, dev_one_hot),
									callbacks=cbfunctions)

		plot_callback(callback_history, args.models_folder)

	elif mode == 'train-relation':

		relationMention_ph = '../resource/data/ace-2005/relationMention/english/data-set/'
		train_data, val_data, test_data = io.load_relation_from_existing_sets(relationMention_ph)

		print("Training data size: {}".format(len(train_data)))
		print("Validation data size: {}".format(len(val_data)))
		print("Testing data size: {}".format(len(val_data)))

		to_one_hot = tf.keras.utils.to_categorical
		graphs_to_indices = my_models.to_indices_with_extracted_entities

		train_as_indices = list(graphs_to_indices(train_data, word2idx))
		print("Dataset shapes: {}".format([d.shape for d in train_as_indices]))

		train_data = None

		n_out = p['relation_type_n']  # n_out = number of relation categories
		print("N_out:", n_out)

		val_as_indices = list(graphs_to_indices(val_data, word2idx))
		# val_data = None

		test_as_indices = list(graphs_to_indices(test_data, word2idx))
		test_data = None

		# sentences_matrix, arg1_matrix, arg2_matrix, y_matrix
		train_y_properties_one_hot = to_one_hot(train_as_indices[-1], n_out)

		val_y_properties_one_hot = to_one_hot(val_as_indices[-1], n_out)

		test_y_properties_one_hot = to_one_hot(test_as_indices[-1], n_out)
		entity_model = models.load_model(args.entity_folder + "model_entity" + "-" + args.metadata + ".kerasmodel")

		entity_weights = entity_model.get_layer(name='entity_BiLSTM_layer').get_weights()

		relation_model = getattr(my_models, model_name)(embeddings, entity_weights)
		callback_history = relation_model.fit(train_as_indices[:-1],
									[train_y_properties_one_hot],
									epochs=args.epoch,
									batch_size=p['batch_size'],
									verbose=1,
									validation_data=(val_as_indices[:-1], val_y_properties_one_hot),
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