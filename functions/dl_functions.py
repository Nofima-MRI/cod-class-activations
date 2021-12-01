import logging

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam, SGD


from tensorflow.keras.applications import VGG16, resnet50
from tensorflow.keras.applications import inception_v3
from tensorflow.keras.applications import xception
from tensorflow.keras.applications import inception_resnet_v2
from tensorflow.keras.applications import densenet

# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession
# config = ConfigProto()
# config.gpu_options.allow_growth = True
# session = InteractiveSession(config = config)

# import tensorflow as tf
# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True
# session = tf.compat.v1.InteractiveSession(config=config)

def get_cnn_model(cnn_type, input_shape, num_classes, learning_rate, loss, metrics = ['accuracy'], optimizer_name = 'adam'):


	pooling  = 'avg'
	# num_classes = 1

	# create cnn model
	if cnn_type == 'vgg16':
		
		# https://www.tensorflow.org/api_docs/python/tf/keras/applications/VGG16
		model = VGG16(include_top = True, weights = None, input_shape = input_shape, pooling = pooling, classes = num_classes)

		# # create new flatten layer and append it to the model output
		# x = Flatten()(model.output)
		# # create new dense layer and append to x
		# x = Dense(1, activation = 'sigmoid')(x)

		# # create new model
		# model = Model(inputs = model.input, outputs = x)

	elif cnn_type == 'vgg16_sequential':

		model = get_VGG16(input_shape = input_shape, num_classes = num_classes)

	elif cnn_type == 'resnet50':

		#https://www.tensorflow.org/api_docs/python/tf/keras/applications/resnet50/ResNet50
		model = resnet50.ResNet50(include_top = True, weights = None, input_shape = input_shape, pooling = pooling, classes = num_classes)
	
	elif cnn_type == 'inception_v3':

		# https://www.tensorflow.org/api_docs/python/tf/keras/applications/inception_v3/InceptionV3
		model = inception_v3.InceptionV3(include_top = True, weights = None, input_shape = input_shape, pooling = pooling, classes = num_classes)

	elif cnn_type == 'xception':

		# https://www.tensorflow.org/api_docs/python/tf/keras/applications/xception/Xception

		model = xception.Xception(include_top = True, weights = None, input_shape = input_shape, pooling = pooling, classes = num_classes)

	elif cnn_type == 'resnet_v2':

		# https://www.tensorflow.org/api_docs/python/tf/keras/applications/inception_resnet_v2/InceptionResNetV2
		model = inception_resnet_v2.InceptionResNetV2(include_top = True, weights = None, input_shape = input_shape, pooling = pooling, classes = num_classes)
    
	elif cnn_type == 'densenet121':

		# https://www.tensorflow.org/api_docs/python/tf/keras/applications/densenet/DenseNet121
		model = densenet.DenseNet121(include_top = True, weights = None, input_shape = input_shape, pooling = pooling, classes = num_classes)

	else:
		logging.error(f'Model type {cnn_type} not implemented')
		exit(1)

	"""
		Define optimizer
	"""
	if optimizer_name == 'adam':
		optimizer = Adam(learning_rate)
	elif optimizer_name == 'sgd':
		optimizer = SGD(learning_rate)
	else:
		logging.error(f'Optimizer {optimizer_name} not yet implemented.')
		exit(1)
	
	# compile model
	model.compile(optimizer = optimizer, loss = loss, metrics = metrics)

	return model


def get_VGG16(input_shape, num_classes):
	model = Sequential()
	model.add(Conv2D(input_shape=input_shape,filters=64,kernel_size=(3,3),padding="same", activation="relu"))
	model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
	model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
	
	model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
	model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
	model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
	
	model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
	model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
	model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
	model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
	
	model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
	model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
	model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
	model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
	
	model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
	model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
	model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
	model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2),name='vgg16'))
	
	model.add(Flatten(name='flatten'))
	model.add(Dense(4096, activation='relu', name='fc1'))
	model.add(Dropout(0.5))
	model.add(Dense(4096, activation='relu', name='fc2'))
	model.add(Dropout(0.5))
	model.add(Dense(4096, activation='relu', name='fc3'))
	model.add(Dropout(0.5))
	model.add(Dense(num_classes, activation='softmax', name='output'))
	return model
	