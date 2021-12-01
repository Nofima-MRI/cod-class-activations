import os
import math
import numpy as np
import logging
import scipy

from sklearn.utils import shuffle
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from functions.helper_functions import create_directory


def upsample_arrays(X, Y, perform_shuffle = True, verbose = True, random_state = 42):

	"""
	1) Calculate instances per class
	"""

	# define all classes
	class_labels  = np.unique(Y)

	# calculate the number of samples within specific class
	total_class_labels = {}
	for class_label in class_labels:

		# count number of class instances and update numpy array
		total_class_labels[class_label] = (Y == class_label).sum()
		
		if verbose:
			# verbose
			logging.info(f'Class {class_label} : {total_class_labels[class_label]}')
	
	"""
	2) Upsample data
	"""

	# empty lists to hold upsampled data
	X_upsamples, Y_upsamples = [], []

	# calculate the class with the highest number of samples
	max_class_samples = max([value for value in total_class_labels.values()])
	# loop over each class
	for class_label in class_labels:
		# the class that has the most instances we skip, since we don't upsample them.
		# also if the class has the same number of samples than the max class, then we also skip this
		if total_class_labels[class_label] < max_class_samples:
			
			# calculate how many random instances we need to copy
			num_upsample = max_class_samples - total_class_labels[class_label]

			if verbose:
				logging.debug(f'Upsamling class {class_label} with {num_upsample} random samples')

			# get class values from array
			y_indexes, _ = np.where(Y == class_label)

			# shuffle indexes and only take num_sample
			y_indexes = shuffle(y_indexes, random_state = random_state)[:num_upsample]

			# get the X data based on the y_indexes (we use y index since we want to have the index belonging to class i)
			x_upsamples = np.take(X, y_indexes, axis = 0)

			# add to list
			X_upsamples.append(x_upsamples)
			# add Y data, this we create dynamically because we know how many samples and we know the class = i
			Y_upsamples.append(np.full((len(y_indexes),1), class_label))
	
	# only upsample if there are values in X_upsamples array
	if len(X_upsamples) > 0:
		
		# combine X_upsamples into single array
		X_upsamples = np.vstack(X_upsamples)
		Y_upsamples = np.vstack(Y_upsamples)

		# combine original X and X_upsamples
		X = np.vstack((X, X_upsamples))
		Y = np.vstack((Y, Y_upsamples))
	else:
		logging.warning('Upsampling not required, found balanced class data')

	"""
	3) Shuffle data
	"""

	# shuffle if set to True
	if perform_shuffle:

		if verbose:
			logging.info('Perform shuffling of the data')
		
		# perform shuffling
		X, Y = shuffle_arrays(X, Y, random_state = random_state)

	return X, Y

def shuffle_arrays(X, Y, random_state = 42):
	"""
	Shuffle X and Y array

	Parameters
	-------------
	X : np.array()
		numpy array with, for example, X features
	Y : np.array()
		numpy array with, for example, Y labels
	random_sate : int (optional)
		seed for randomness

	Returns
	-----------
	X : np.array()
		numpy array with shuffled rows in the same way as Y
	Y : np.array()
		numpy array with shuffled rows in the same way as X
	"""

	# shuffle all X and Y
	X, Y = shuffle(X, Y, random_state = random_state)

	return X, Y

def get_train_val_test_datasets(X, Y, train_split, val_split, perform_shuffle = True, random_state = 42):
	"""
		split X and Y arrays into training, development, and test
	"""	

	if perform_shuffle:
		X, Y = shuffle_arrays(X, Y, random_state = random_state)

	# calculate size of training set
	train_size = int(X.shape[0] * train_split)
	# calculate size of validation set (remaining is left for test set)
	val_size = int(X.shape[0] * val_split)

	# create train, dev, test set
	datasets = {'X_train' : X[:train_size],
				'X_val' : X[train_size:train_size + val_size],
				'X_test' : X[train_size + val_size:],
				'Y_train' : Y[:train_size],
				'Y_val' : Y[train_size:train_size + val_size],
				'Y_test' : Y[train_size + val_size:]}

	return datasets

def create_image_data_generator(x, y, batch_size, rescale = None, rotation_range = None, width_shift_range = None, height_shift_range = None, 
								shear_range = None, zoom_range = None, horizontal_flip = None, vertical_flip = None, brightness_range = None,
								save_to_dir = None, seed = 42):
	"""
	Create image data generator for tensorflow

	Parameters
	------------
	x : np.ndarray or os.path
		X features. Either direct as numpy array or as os.path which will then be loaded
	y : np.ndarray or os.path
		Y labels. Either direct as numpy array or as os.path which will then be loaded
	"""

	# create image data generator
	img_args = {}

	# convert arguments to dictionary when not None
	if rescale is not None:
		img_args['rescale'] = rescale
	if rotation_range is not None:
		img_args['rotation_range'] = rotation_range
	if width_shift_range is not None:
		img_args['width_shift_range'] = width_shift_range
	if height_shift_range is not None:
		img_args['height_shift_range'] = height_shift_range
	if shear_range is not None:
		img_args['shear_range'] = shear_range
	if zoom_range is not None:
		img_args['zoom_range'] = zoom_range
	if horizontal_flip is not None:
		img_args['horizontal_flip'] = horizontal_flip
	if vertical_flip is not None:
		img_args['vertical_flip'] = vertical_flip
	if brightness_range is not None:
		img_args['brightness_range'] = brightness_range

	# create save_to_dir folder if not None
	if save_to_dir is not None:
		create_directory(save_to_dir)
			
	# create ImageDataGenerator from unpacked dictionary
	image_data_generator = ImageDataGenerator(**img_args)


	# check if x is numpy array, if not, then load x
	x = x if type(x) is np.ndarray or type(x) is np.ma.core.MaskedArray else np.load(x)
	# same for y
	y = y if type(y) is np.ndarray else np.load(y)

	# create the generator
	generator = image_data_generator.flow(x = x, y = y, batch_size = batch_size, seed = seed, save_to_dir = save_to_dir)

	return generator

def mean_confidence_interval(data, confidence=0.95):
	a = 1.0 * np.array(data)
	n = len(a)
	m, se = np.mean(a), scipy.stats.sem(a)
	h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
	return m, m-h, m+h, h

def calculate_ci_interval(value, n, ci = 95):
	"""
	Calculate 95% confidence interval

	Parameters
	----------
	value : float
		value for which the ci needs to be calculated
	n : int
		number of samples
	ci : string
		95% confidence interval
	"""
	
	# convert confidence level to z-score
	ci_to_z = {	90 : 1.645,
				95 : 1.960,
				98 : 2.326,
				99 : 2.576}

	# get right 
	z = ci_to_z.get(ci)

	# check for null
	if z is None:
		logging.error(f'Confidence level {ci} not implemented')
		exit()

	# calculate confidence interval
	ci =  z * math.sqrt( (value * (1 - value)) / n)

	return ci