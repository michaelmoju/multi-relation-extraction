import word_embeddings
from miwaIO.instances_get import *
import my_models
import tensorflow as tf
from tensorflow.python.keras import callbacks
import matplotlib.pyplot as plt


def prediction(model, data_input):
	predictions = model.predict(data_input, batch_size=my_models.model_params['batch_size'], verbose=1)

	predictions_classes = np.argmax(predictions, axis=-1)

	return predictions_classes


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


if __name__ == '__main__':
	import argparse

	parser = argparse.ArgumentParser()
	parser.add_argument('model_name')
	parser.add_argument('mode', choices=['train-entity', 'train-relation', 'evaluate', 'predict'])
	parser.add_argument('--epoch', default=10)
	parser.add_argument('--data_path', default='../resource/data/ace-2005/miwa2016/corpus/')
	parser.add_argument('--embedding', default='../resource/embeddings/glove/glove.6B.50d.txt')
	parser.add_argument('--metadata', default='10', type=str)
	parser.add_argument('--checkpoint', action='store_true')
	parser.add_argument('--models_folder', default="./trainedmodels/")
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

	if mode == 'train-entity':
		model = getattr(my_models, model_name)(embeddings)

		cbfunctions = []
		if args.checkpoint:
			checkpoint = callbacks.ModelCheckpoint(args.models_folder + model_name + '-{epoch:02d}' + ".kerasmodel",
				monitor='val_loss', verbose=1, save_best_only=True)
			cbfunctions.append(checkpoint)

		callback_history = model.fit(train_sent, train_one_hot, epochs=args.epoch, batch_size=model_params["batch_size"],
									 validation_data=(dev_sent, dev_one_hot),
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