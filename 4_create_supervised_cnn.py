import os
import re
import numpy as np
import pandas as pd

from functions.helper_functions import set_start, get_current_timestamp, get_current_timestamp, create_directory, read_directory
from functions.project_functions import get_paths, get_parameters, create_patients, get_datasets_paths, check_mri_slice_validity, get_mri_image_group_name
from functions.hdf5_functions import read_dataset_from_group, read_metadata_from_group_dataset
from functions.data_functions import upsample_arrays, get_train_val_test_datasets, create_image_data_generator
from functions.dl_functions import get_cnn_model

from tensorflow import distribute
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

def create_train_val_test_datasets(paths, params):
	"""
	Read in X and Y features and create a training, development, and test set

	Parameters
	-----------
	paths : dict()
		folder locations to read/write data to (see project_functions)
	params : dict()
		dictionary holding project wide parameter values
	"""

	# create list of patient names for filtering
	patients = create_patients(treatments = params['limit_treatments'], samples = params['limit_samples'], states = params['limit_states'])

	"""
		Get images and class labels
	"""
	
	# empty lists to hold X features and Y labels
	X = []
	Y = []

	# dynamically create hdf5 file
	hdf5_file = os.path.join(paths['hdf5_folder'], params['hdf5_file'])

	# read images, and create cropped image
	for i, patient in enumerate(patients):

		logging.info(f'Processing patient : {patient} {i}/{len(patients)}')

		# read original MRI image as input feature
		x = read_dataset_from_group(group_name = get_mri_image_group_name(), dataset = patient, hdf5_file = hdf5_file)

		# check which slices are valid
		valid_slices = [check_mri_slice_validity(patient = patient, mri_slice = mri_slice, total_num_slices = x.shape[0]) for mri_slice in range(x.shape[0])]
		
		# only take valid slices
		x = x[valid_slices]

		# expand X dimensions to add channel
		x = np.expand_dims(x, 3)

		# read in meta data
		meta_data = read_metadata_from_group_dataset(group_name = params['group_original_mri'], dataset = patient, hdf5_file = hdf5_file)

		# read class label
		class_label = meta_data['ClassLabel']

		# create y vector 
		y = np.full(shape = (x.shape[0],1), fill_value = class_label)
	
		# add x to X and y to Y
		X.append(x)
		Y.append(y)

	# convert list to numpy array
	X = np.vstack(X)
	Y = np.vstack(Y)

	logging.info(f'X shape : {X.shape}')
	logging.info(f'Y shape : {Y.shape}')

	# let y labels start from zero instead of some other class ID
	if params['start_y_from_zero']:

		# calulate how much y need to shift to start from zero
		min_y = np.min(Y)

		logging.info(f'Starting Y from zero, shifting y by: {min_y}')
		# let Y start from zero
		Y = Y - min_y

	# create binary dataset by setting all frozen/thawed class labels to the same class
	if params['create_binary_dataset']:

		logging.info('Creating binary dataset')
		# lowest class label remains the same but any other class is set to the same class
		min_y = np.min(Y)
		# set all other classes than fresh (which is the lowest class) to the same class label
		Y[Y != min_y] = min_y + 1
	
	# upsample data if set to True
	if params['upsample']:

		logging.info('Start upsampling')
		# upsample underrepresented samples to create a class balanced dataset
		X, Y = upsample_arrays(X, Y, perform_shuffle = True)

		logging.info(f'X shape : {X.shape}')
		logging.info(f'Y shape : {Y.shape}')

	# create datasets
	datasets = get_train_val_test_datasets(X, Y, train_split = params['train_split'], val_split = params['val_split'])

	# dynamically create dataset subfolder
	dataset_folder = os.path.join(paths['dataset_folder'], get_mri_image_group_name())
	# create folder
	create_directory(dataset_folder)

	# save each dataset to disk
	for label, dataset in datasets.items():

		logging.debug(f'{label} shape {dataset.shape}')

		# save as numpy array to file
		np.save(os.path.join(dataset_folder, label), dataset)

def train_cnn_classifier(paths, params):
	"""
	Train CNN classifier

	Parameters
	-----------


	"""

	# folder where to save datasets to
	dataset_folder = os.path.join(paths['dataset_folder'], get_mri_image_group_name())

	# type of architecture to 
	cnn_architectures = ['resnet50', 'inception_v3', 'xception', 'resnet_v2', 'densenet121', 'vgg16']

	for cnn_architecture in cnn_architectures:

		# read datasets from file
		datasets = get_datasets_paths(dataset_folder)

		# read one dataset and extract number of classes
		num_classes = len(np.unique(np.load(datasets['Y_train'])))
		# read input shape
		input_shape = np.load(datasets['X_train']).shape

		# model checkpoint and final model save folder
		model_save_folder = os.path.join(paths['model_folder'], get_mri_image_group_name(), f'{get_current_timestamp()}_{cnn_architecture}')
		# create folder
		create_directory(model_save_folder)

		"""
			MASKING INPUT IMAGE
		"""
		# # mask background of input features
		# if params['mask_background']:
		# 	# loop over dictionary
		# 	for key, value in datasets.items():
		# 		# load numpy array as set as new value
		# 		if key in ['X_train', 'X_val', 'X_test']:
		# 			datasets[key] = np.ma.masked_equal(x = np.load(value), value = 0)		
					
		# 		else:
		# 			datasets[key] = np.load(value)
					
		"""
			DEFINE LEARNING PARAMETERS
		"""
		params.update({'ARCHITECTURE' : cnn_architecture,
					'NUM_CLASSES' : num_classes,
					'LR' : .005,
					'OPTIMIZER' : 'sgd',
					'INPUT_SHAPE' : input_shape[1:],
					'BATCH_SIZE' : 32,
					'EPOCHS' : 2000,
					'ES' : True,
					'ES_PATIENCE' : 250,
					'ES_RESTORE_WEIGHTS' : True,
					'SAVE_CHECKPOINTS' : True,
					'RESCALE' : params['rescale_factor'],
					'ROTATION_RANGE' : 45,
					'WIDTH_SHIFT_RANGE' : 0.2,
					'HEIGHT_SHIFT_RANGE' : 0.2,
					'SHEAR_RANGE' : None,
					'ZOOM_RANGE' : None,
					'HORIZONTAL_FLIP' : True,
					'VERTICAL_FLIP' : True,
					'BRIGHTNESS_RANGE' : None,
					})
		"""
			DATAGENERATORS
		"""

		# file name of X training data
		x_train_file_name = 'X_train' if not params['use_grayscale_augmented_images_for_training'] else 'X_train_aug'
		y_train_file_name = 'Y_train' if not params['use_grayscale_augmented_images_for_training'] else 'Y_train_aug'

		# generator for training data
		train_generator = create_image_data_generator(x = datasets[x_train_file_name], y = datasets[y_train_file_name], batch_size = params['BATCH_SIZE'], rescale = params['RESCALE'],
													rotation_range = params['ROTATION_RANGE'], width_shift_range = params['WIDTH_SHIFT_RANGE'],
													height_shift_range = params['HEIGHT_SHIFT_RANGE'], shear_range = params['SHEAR_RANGE'], 
													zoom_range = params['ZOOM_RANGE'], horizontal_flip = params['HORIZONTAL_FLIP'],
													vertical_flip = params['VERTICAL_FLIP'], brightness_range = params['BRIGHTNESS_RANGE'],
													save_to_dir = None if paths['augmentation_folder'] is None else paths['augmentation_folder'])

		# generator for validation data
		val_generator = create_image_data_generator(x = datasets['X_val'], y = datasets['Y_val'], batch_size = params['BATCH_SIZE'], rescale = params['RESCALE'])	
		
		# generator for test data
		test_generator = create_image_data_generator(x = datasets['X_test'], y = datasets['Y_test'], batch_size = params['BATCH_SIZE'], rescale = params['RESCALE'])	

		"""
			CALLBACKS
		"""

		# empty list to hold callbacks
		callback_list = []

		# early stopping callback
		if params['ES']:
			callback_list.append(EarlyStopping(monitor = 'val_loss', min_delta = 0, patience = params['ES_PATIENCE'], restore_best_weights = params['ES_RESTORE_WEIGHTS'], verbose = 1, mode = 'auto'))

		# save checkpoints model
		if params['SAVE_CHECKPOINTS']:
			# create checkpoint subfolder
			create_directory(os.path.join(model_save_folder, 'checkpoints'))
			callback_list.append(ModelCheckpoint(filepath = os.path.join(model_save_folder, 'checkpoints', 'checkpoint_model.{epoch:02d}_{val_loss:.2f}_{val_accuracy:.2f}.h5'), save_weights_only = False, monitor = 'val_loss', mode = 'auto', save_best_only = True))

		"""
			TRAIN CNN MODEL
		"""
		
		# METRICS = [	
		# 			metrics.TruePositives(name='tp'),
		# 			metrics.FalsePositives(name='fp'),
		# 			metrics.TrueNegatives(name='tn'),
		# 			metrics.FalseNegatives(name='fn'), 
		# 			metrics.BinaryAccuracy(name='accuracy'),
		# 			metrics.Precision(name='precision'),
		# 			metrics.Recall(name='recall'),
		# 			metrics.AUC(name='auc')
		# 			]
	
		METRICS = ['accuracy']

		# use multi GPUs
		mirrored_strategy = distribute.MirroredStrategy()
		
		# context manager for multi-gpu
		with mirrored_strategy.scope():

			# define loss function
			# loss = losses.BinaryCrossentropy()
			# loss = losses.CategoricalCrossentropy()
			loss = 'sparse_categorical_crossentropy'
			# loss = 'categorical_crossentropy'
		
			# get cnn model architecture
			model = get_cnn_model(cnn_type = params['ARCHITECTURE'], 
								input_shape = params['INPUT_SHAPE'], 
								num_classes = params['NUM_CLASSES'], 
								learning_rate = params['LR'], 
								optimizer_name = params['OPTIMIZER'],
								loss = loss,
								metrics = METRICS)

			history = model.fit(train_generator,
				epochs = params['EPOCHS'], 
				steps_per_epoch = len(train_generator),
				validation_data = val_generator,
				validation_steps = len(val_generator),
				callbacks = callback_list)

			# evaluate on test set
			history_test = model.evaluate(test_generator)

			# save the whole model
			model.save(os.path.join(model_save_folder, 'model.h5'))
			
			# save history of training
			pd.DataFrame(history.history).to_csv(os.path.join(model_save_folder, 'history_training.csv'))
			
			# save test results
			# pd.DataFrame(history_test, index = ['test_loss', 'test_tp', 'test_fp', 'test_tn', 'test_fn', 'test_accuracy', 'test_precision', 'test_recall', 'test_auc']).to_csv(os.path.join(model_save_folder, 'history_test.csv'))
			pd.DataFrame(history_test, index = ['test_loss', 'test_accuracy']).to_csv(os.path.join(model_save_folder, 'history_test.csv'))
			# save model hyperparameters
			pd.DataFrame(pd.Series(params)).to_csv(os.path.join(model_save_folder, 'params.csv'))

def calculate_model_performance(paths, params):
	"""
	Calculate classification performance of CNN model
	"""

	# create list of patient names for filtering
	patients = create_patients(treatments = params['limit_treatments'], samples = params['limit_samples'], states = params['limit_states'])

	# dynamically create hdf5 file
	hdf5_file = os.path.join(paths['hdf5_folder'], params['hdf5_file'])

	# append dataset name to model folder
	model_folder = os.path.join(paths['model_folder'], get_mri_image_group_name())

	# read model subfolders to process
	model_subfolders = [x for x in read_directory(directory = model_folder, subfolders = False) if x != '.DS_Store']

	# overwrite for testing
	model_subfolders = ['20210630133923_resnet50'] # 4 class model
	# model_subfolders = ['20210629152135_resnet50'] # 2 class model
	
	# process each folder
	for model_subfolder in model_subfolders:

		logging.info(f'Processing model subfolder {model_subfolder}')

		# load model or load latest checkpoint model (should be the same if best model is restored during training)
		if not params['use_checkpoint_model']:
			# load cnn model
			model = load_model(os.path.join(model_folder, model_subfolder, 'model.h5'))
		else:

			# hold checkpoint file locations with epoch number
			checkpoint_model_files = []
			# read last checkpoint model
			for x in read_directory(os.path.join(model_folder, model_subfolder, 'checkpoints')):
				# pars out epoch number
				epoch = re.search(r'checkpoint_model\.([0-9]+)_', x.split(os.sep)[-1]).group(1)
				# add to tuple
				checkpoint_model_files.append([int(epoch), x])
			
			# get highest epoch number
			checkpoint_model_file = sorted(checkpoint_model_files, key = lambda tup : tup[0], reverse = True)[0][1]
			
			# load checkpoint model
			model = load_model(checkpoint_model_file)


		# loop over patient to calculate performance
		for i, patient in enumerate(patients):

			# create empty dataframe
			df = pd.DataFrame()
  
			logging.info(f'Processing patient : {patient} {i}/{len(patients)}')

			# read original MRI image as input feature
			x = read_dataset_from_group(group_name = get_mri_image_group_name(), dataset = patient, hdf5_file = hdf5_file)

			# check which slices are valid
			valid_slices = np.array([check_mri_slice_validity(patient = patient, mri_slice = mri_slice, total_num_slices = x.shape[0]) for mri_slice in range(x.shape[0])])

			# expand X dimensions to add channel
			x = np.expand_dims(x, 3)

			# rescale image
			x = x * params['rescale_factor']

			# read in meta data
			meta_data = read_metadata_from_group_dataset(group_name = params['group_original_mri'], dataset = patient, hdf5_file = hdf5_file)

			# read class label
			class_label = meta_data['ClassLabel']

			# create y vector 
			y = np.full(shape = (x.shape[0]), fill_value = class_label)

			# create binary dataset by setting all frozen/thawed class labels to the same class
			if params['create_binary_dataset']:
				# set all other classes than fresh (which is the lowest class) to the same class label
				y[y != 0] = 1

			# perform inference of cnn model
			y_hat = model.predict(x)
			# convert to 1 class
			y_hat = np.argmax(y_hat, axis = 1)

			# perform inference of cnn model
			# y_hat = model.predict_classes(x)

			# loop over each y and y_hat
			for ii in range(y.shape[0]):
				# check if slice is valid
				if valid_slices[ii]:

					# create new dataframe from seriers
					df_row = pd.DataFrame([pd.Series({'patient' : patient, 'mri_slice' : ii, 'y' : y[ii], 'y_hat' : y_hat[ii]})])
					
					# then append that dataframe to overall dataframe
					df = pd.concat([df, df_row], ignore_index=True)
			
			# save results of each patient
			save_folder = os.path.join(paths['table_folder'], 'classification_results', get_mri_image_group_name(), model_subfolder)
			# create that folder
			create_directory(save_folder)
			# save dataframe to folder
			df.to_csv(os.path.join(save_folder, f'{patient}.csv'))


def add_contrast_augmentation_training_data(paths, params):

	# groups that need to be added
	groups = ['bg_True_crop_False_gray_True_adap_histogram',
				'bg_True_crop_False_gray_True_contrast',
				'bg_True_crop_False_gray_True_histogram']

	# create list of patient names for filtering
	patients = create_patients(treatments = params['limit_treatments'], samples = params['limit_samples'], states = params['limit_states'])
	# dynamically create hdf5 file
	hdf5_file = os.path.join(paths['hdf5_folder'], params['hdf5_file'])

	"""
		Get images and class labels
	"""
	
	# empty lists to hold X features and Y labels
	X = []
	Y = []

	for group in groups:

		logging.info(f'Processing group {group}')
			
		# read images, and create cropped image
		for i, patient in enumerate(patients):

			logging.info(f'Processing patient : {patient} {i}/{len(patients)}')

			# read original MRI image as input feature
			x = read_dataset_from_group(group_name = get_mri_image_group_name(), dataset = patient, hdf5_file = hdf5_file)

			# check which slices are valid
			valid_slices = [check_mri_slice_validity(patient = patient, mri_slice = mri_slice, total_num_slices = x.shape[0]) for mri_slice in range(x.shape[0])]
			
			# only take valid slices
			x = x[valid_slices]

			# expand X dimensions to add channel
			x = np.expand_dims(x, 3)

			# read in meta data
			meta_data = read_metadata_from_group_dataset(group_name = params['group_original_mri'], dataset = patient, hdf5_file = hdf5_file)

			# read class label
			class_label = meta_data['ClassLabel']

			# create y vector 
			y = np.full(shape = (x.shape[0],1), fill_value = class_label)
		
			# add x to X and y to Y
			X.append(x)
			Y.append(y)
	
	# convert list to numpy array
	X = np.vstack(X)
	Y = np.vstack(Y)

	# let y labels start from zero instead of some other class ID
	if params['start_y_from_zero']:
		# calulate how much y need to shift to start from zero
		min_y = np.min(Y)
		# let Y start from zero
		Y = Y - min_y

	# create binary dataset by setting all frozen/thawed class labels to the same class
	if params['create_binary_dataset']:
		# lowest class label remains the same but any other class is set to the same class
		min_y = np.min(Y)
		# set all other classes than fresh (which is the lowest class) to the same class label
		Y[Y != min_y] = min_y + 1
	
	# upsample data if set to True
	if params['upsample']:

		logging.info('Start upsampling')
		# upsample underrepresented samples to create a class balanced dataset
		X, Y = upsample_arrays(X, Y, perform_shuffle = True)

		logging.info(f'X shape : {X.shape}')
		logging.info(f'Y shape : {Y.shape}')


	logging.info(f'X shape : {X.shape}')
	logging.info(f'Y shape : {Y.shape}')

	# add original training data to X and Y
	X_train = np.load(os.path.join(paths['dataset_folder'], get_mri_image_group_name(), 'X_train.npy'))
	Y_train = np.load(os.path.join(paths['dataset_folder'], get_mri_image_group_name(), 'Y_train.npy'))

	# combine X and X_train
	X = np.vstack([X, X_train])
	Y = np.vstack([Y, Y_train])

	logging.info(f'X shape : {X.shape}')
	logging.info(f'Y shape : {Y.shape}')

	# upsample data if set to True
	if params['upsample']:

		logging.info('Start upsampling')
		# upsample underrepresented samples to create a class balanced dataset
		X, Y = upsample_arrays(X, Y, perform_shuffle = True)

		logging.info(f'X shape : {X.shape}')
		logging.info(f'Y shape : {Y.shape}')

	# save as numpy array to file
	np.save(os.path.join(paths['dataset_folder'], get_mri_image_group_name(), 'X_train_aug.npy'), arr = X)
	np.save(os.path.join(paths['dataset_folder'], get_mri_image_group_name(), 'Y_train_aug.npy'), arr = Y)

if __name__ == "__main__":

	tic, process, logging = set_start()

	# get all folder paths
	paths = get_paths()
	# get project parameters
	params = get_parameters()

	# 1) create training, validation, and test dataset
	# create_train_val_test_datasets(paths = paths, params = params)

	# 2) add images that have been contrast adjusted to training dataset
	add_contrast_augmentation_training_data(paths = paths, params = params)

	# 2) Train CNN model
	# train_cnn_classifier(paths = paths, params = params)

	# 3) Calculate model performance
	# calculate_model_performance(paths = paths, params = params)

