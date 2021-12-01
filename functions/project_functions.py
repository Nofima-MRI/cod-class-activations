"""
Project functions are functions that are specifically written for this project and typically find no use within another project

Author : Shaheen Syed
Date created : 2020-08-28

"""
import os
import re
import logging
import socket
import pandas as pd
import numpy as np

from functions.helper_functions import create_directory, read_directory
from functions.hdf5_functions import read_dataset_from_group

def get_environment():

	# local environment should be: Shaheens-MacBook-Pro-2.local
	# nofima GPU workstation should be: shaheensyed-gpu
	# UIT GPU workstation should be: shaheengpu
	return socket.gethostname()

def get_paths(env = None, create_folders = True):
	"""
	Get all project paths
	"""

	# if environement argument is not given then get hostname with socket package
	if env is None:
		env = get_environment()

	# empty dictionary to return
	paths = {}

	# name of the project
	# paths['project_name'] = 'cod_class_activations'
	paths['project_name'] = 'cod_class_activations'

	# path for local machine
	if env in ['Shaheens-MacBook-Pro-2.local', 'shaheens-mbp-2.lan', 'dhcp3803-int-ans.vpn.uit.no']:
		# project base folder
		paths['base_path'] = os.path.join(os.sep, 'Users', 'shaheen.syed', 'data', 'projects', paths['project_name'])
	elif env == 'shaheensyed-gpu':
		# base folder on nofima GPU workstation
		paths['base_path'] = os.path.join(os.sep, 'home', 'shaheensyed', 'projects', paths['project_name'])
	elif env == 'shaheengpu':
		# base folder on UIT GPU workstation
		paths['base_path'] = os.path.join(os.sep, 'home', 'shaheen', 'projects', paths['project_name'])
	else:
		logging.error(f'Environment {env} not implemented.')
		exit(1)

	# folder contained original MRI data in Dicom format
	paths['mri_folder'] = os.path.join(paths['base_path'], 'data', 'mri')
	# folder for HDF5 files
	paths['hdf5_folder'] = os.path.join(paths['base_path'], 'data', 'hdf5')
	# folder for .dcm files with new patient name
	paths['dcm_folder'] = os.path.join(paths['base_path'], 'data', 'dcm')
	# folder for datasets
	paths['dataset_folder'] = os.path.join(paths['base_path'], 'data', 'datasets')
	# folde for trained models
	paths['model_folder'] = os.path.join(paths['base_path'], 'models')
	# define folder for data augmentation
	paths['augmentation_folder'] = None#os.path.join(paths['base_path'], 'data', 'augmentation')
	# folder for class activations
	paths['class_activation_folder'] = os.path.join(paths['base_path'], 'data', 'class_activations')
	# folder for plots
	paths['plot_folder'] = os.path.join(paths['base_path'], 'plots')
	# folder for paper plots
	paths['paper_plot_folder'] = os.path.join(paths['base_path'], 'paper_plots')
	# folder for tables
	paths['table_folder'] = os.path.join(paths['base_path'], 'data', 'tables')
	# folder that holds supervised classification data
	paths['supervised_folder'] = os.path.join(paths['base_path'], 'data', 'supervised', 'original')
	# folder that holds supervised classification data that is smoothed with a kernel
	paths['supervised_smoothed_folder'] = os.path.join(paths['base_path'], 'data', 'supervised', 'smoothed')

	# create all folders if not exist
	if create_folders:
		for folder in paths.values():
			if folder is not None:
				if folder != paths['project_name']:
					create_directory(folder)	
			
	return paths

def get_parameters():
	"""
	Project parameters
	"""

	# empty dictionary to hold parameters
	params = {}

	# HDF5 file name
	params['hdf5_file'] = 'FISH-MRI.h5'
	# original MRI group name
	params['group_original_mri'] = 'ORIGINAL-MRI'

	# use images with background removed
	params['use_no_bg_images'] = True
	# use cropped imaged
	params['use_cropped_images'] = False
	# dimensions for subset (patch) of the image, ignored if use_cropped_images is False
	params['crop_dimensions'] = (128,128)
	# dynamic midpoint to get the middle of the sample scan, ignored if use_cropped_images is False
	params['dynamic_midpoint'] = True
	# 

	"""
	Grayscale adjustment
	"""
	# adjust gray scale
	params['adjust_gray_scale'] = False
	# grayscale adjustment method ['contrast' | 'histogram' | 'adap_histogram']
	params['gray_scale_method'] = 'adap_histogram'
	# if adjust_gray_scale method is contrast, then use following lower bound percentile
	params['contrast_percentile_lower_bound'] = 0.1
	# if adjust_gray_scale method is contrast, then use following upper bound percentile
	params['contrast_percentile_upper_bound'] = 99.9

	"""
		Dataset parameters
	"""
	# size of the training data
	params['train_split'] = 0.8
	# size of the validation data 
	params['val_split'] = 0.1
	# limit treatments, can be [1,2,3]
	params['limit_treatments'] = []
	# limit samples, can be 1 to 11 
	params['limit_samples'] = [] 
	# limit states, can be fresh = 'fersk' or frozen = 'Tint'
	params['limit_states'] = []
	# let y start from zero, and not from 1 for example since annotations can start at 1
	params['start_y_from_zero'] = True
	# create two classes instead of all classes. Two classes bascially means fresh and frozen/thawed without taking into account the temp. of freezing
	params['create_binary_dataset'] = True
	# upsample unrepresented classes 
	params['upsample'] = True
	# create mask for background
	params['mask_background'] = True
	# use grayscale adjusted images as part of training data
	params['use_grayscale_augmented_images_for_training'] = True
	
	# skip first number of slices due to artifacts
	params['trim_start_slices'] = 15
	# skip last number of slices due to artifacts
	params['trim_end_slices'] = 15
	# rescale factor. Scale 12 bit images to 0 and 1
	params['rescale_factor'] = 1 / (2 ** 12)

	# set non-damaged connected tissue to non-damaged tissue
	params['set_non_damaged_connected_to_non_damaged'] = True
	# set damaged connected tissue to damaged tissue
	params['set_damaged_connected_to_damaged'] = True
	# smoothing kernels for supervised classification
	params['smoothing_kernels'] = [1, 4, 8, 12, 16, 20, 24, 28, 32]
	# define cam threshold
	params['cam_thresholds'] = [x / 10 for x in range(1,10)]
	

	"""
	Class activations
	"""
	params['use_checkpoint_model'] = True
	

	return params

def get_protocol_translation(protocolname):

	# translation of protocolname
	protocol_translation = {'FSE T2w (axial,n)' : 'axial', 'FSE T2w (cor,n)' : 'cor'}

	return protocol_translation.get(str(protocolname))

def create_class_label(patientname):
	"""
	Based on the patientname, create a class label
	"""

	# convert patientname to string
	patientname = str(patientname)

	# create class label based on patientname
	if 'fersk' in patientname:
		return 0
	elif re.search(r'Torsk 1-.*Tint', str(patientname)):
		return 1
	elif re.search(r'Torsk 2-.*Tint', str(patientname)):
		return 2
	elif re.search(r'Torsk 3-.*Tint', str(patientname)):
		return 3
	else:
		logging.error(f'Patientname cannot be converted to class: {patientname}')
		exit(1)

def check_validity_mri_scan(patientname, datetime):
	"""
	Some scans were not successful and need to be removed from the analays

	Torsk 1-1 fersk 20200817101346

	Torsk 1-1 Tint 20200824111933
	Torsk 1-2 Tint 20200824114015
	Torsk 1-10 Tint 20200824130944
	Torsk 2-1 Tint 20200824133947

	"""	

	# concatenate name and datetime
	patientname = f'{patientname} {datetime}'

	invalid_scans = ['Torsk 1-1 fersk 20200817101346',
					'Torsk 1-1 Tint 20200824111933',
					'Torsk 1-2 Tint 20200824114015',
					'Torsk 1-10 Tint 20200824130944',
					'Torsk 2-1 Tint 20200824133947']

	return False if patientname in invalid_scans else True

def create_patients(treatments = [], samples = [], states = []):
	"""
		Create list of patients to process
	"""

	# define treatments
	treatments = range(1,4) if not treatments else treatments
	# define samples
	samples = range(1,12) if not samples else samples
	# define states
	states = ['fersk', 'Tint'] if not states else states

	# empty list to store patients to
	patients = []

	# dynamically create patients
	for treatment in treatments:
		for sample in samples:
			# sample 3-11 does not exist, so we need to skip it. We only have 11 samples for treatment 1 and 2. Treatment 3 has only 10 samples
			if treatment == 3 and sample == 11:
				continue

			for state in states:
				# create patient
				patient = f'Torsk {treatment}-{sample} {state}'
				# add patient to list
				patients.append(patient)


	return patients

def get_datasets_paths(datasets_folder):
	"""
	Read datasetes from datafolder and return back as dictionary of file locations

	Parameters
	----------
	data_folder : os.path
		location of X_train, Y_train, X_val, Y_val, X_test, and Y_test

	Returns
	---------
	datasets : dict()
		Dictionary with for example, X_train as key, and the file location as value
	"""

	# empty dictionary to store data to
	datasets = {}

	# loop over each file in directory and add file name as key and file location as value
	for file in read_directory(datasets_folder):

		# extra dataset name from file
		dataset_name = file.split(os.sep)[-1][:-4]
		# add to dictionary
		datasets[dataset_name] = file

	return datasets

def get_range_slices(mri_data):
	"""
	Get a range of slices that will be included in the analysis. It takes all the slices of an MRI scan, let's say 54, but it trims the start and end which are 
	defined in params
	"""

	params = get_parameters()

	# total number of slices available
	num_slices = mri_data.shape[0]
	
	# define the slices to plot (here we skip the first 'trim_start_slices' and we exclude the last 'trim_end_slices')
	range_slices = range(num_slices)[params['trim_start_slices']:None if params['trim_end_slices'] == 0 else -params['trim_end_slices']]

	return range_slices

def parse_patientname(patient_name):
	"""
	Parse out treatment, sample, and state from patient name

	Parameters
	----------
	patient_name : string
		name of patient to extract treatment, sample and state from

	Returns
	--------
	treatment : string
		1 = -5 freezing, 2 = -20 freezing, 3 = -40 freezing
	sample : string
		group 1, 2, or 3
	state : string
		fersk = fresh/not frozen, Tint = frozen/thawed to specific freezing protocol, see treatment
	"""

	# extract treatment, sample, and state from patient name
	treatment = int(re.search('.* ([0-9]){1}', patient_name).group(1))
	sample = int(re.search('.*-([0-9]{1,2})', patient_name).group(1))
	state = re.search('.*(fersk|Tint)', patient_name).group(1)

	return treatment, sample, state

def check_mri_slice_validity(patient, mri_slice, total_num_slices):
	"""
	When making MRI scans some slices are not valid and should be taken out of the analysis. After a visual inspection, these slices
	should be filtered out

	Parameters
	----------
	patient : string
		patient ID, for example Torks 1-1 fersk
	mri_slice : int
		slice of MRI scan
	"""

	# get parameters
	params = get_parameters()

	# trim start slices
	trim_start_slices = params['trim_start_slices']
	# trim end slices
	trim_end_slices = params['trim_end_slices']
	# skip slices
	valid_range_slices = range(total_num_slices)[trim_start_slices:None if trim_end_slices == 0 else -trim_end_slices]

	# check if whole patient is invalid
	# invalid_patients = ['Torsk 1-4 fersk']
	# if patient in invalid_patients:
	# 	return False

	# check if slice is in valid_range_slices
	if mri_slice not in valid_range_slices:
		return False

	# check if there are other slices that are invalid
	invalid_slices = {
					'Torsk 1-3 Tint' : [10,11],
					'Torsk 1-9 fersk' : range(0,21),
					'Torsk 1-10 Tint' : [10,11,13],
					'Torsk 2-1 Tint' : [10],
					'Torsk 2-3 Tint' : [10,11],
					'Torsk 2-5 fersk' : [10,11],
					'Torsk 2-5 Tint' : range(0,22),
					'Torsk 2-7 Tint' : range(0,15),
					'Torsk 2-9 fersk' : [11],
					'Torsk 2-9 Tint' : range(0,16),
					'Torsk 3-5 Tint' : range(0,18),
					'Torsk 3-6 Tint' : range(0,20)}
	
	if patient in invalid_slices:
		if mri_slice in invalid_slices[patient]:
			return False

	# all other cases the slice is valid
	return True

def get_mri_image_group_name():

	params = get_parameters()

	"""
	dynamically create group name based on the following parameters
	params['use_no_bg_images']
	params['use_cropped_images']
	params['adjust_gray_scale']
	params['gray_scale_method']
	"""

	group_name = f"bg_{params['use_no_bg_images']}_"\
					f"crop_{params['use_cropped_images']}_"\
					f"gray_{params['adjust_gray_scale']}"

	# add gray scale method if gray scale is set to true
	group_name += f"_{params['gray_scale_method']}" if params['adjust_gray_scale'] else ''
	
	return group_name

def get_dataset_name():

	params = get_parameters()

	# name of dataset to use
	params['dataset_subfolder_name'] = f"{params['dataset_name']}_{get_mri_image_group_name()}"

	return params['dataset_subfolder_name']


def treatment_to_title(treatment):
	"""
	Convert treatment ID to human readable title
	"""

	# first check state, since treatment has not yet been done if state is fresh
	titles = {	1 : u'-5 °C Freezing',
				2 : u'-20 °C Freezing',
				3 : u'-40 °C Freezing',
	}
	
	return titles[treatment]

def treatment_to_group_title(treatment):
	"""
	Convert treatment ID to human readable title
	"""

	# first check state, since treatment has not yet been done if state is fresh
	titles = {	1 : u'Group 1 (-5 °C)',
				2 : u'Group 2 (-20 °C)',
				3 : u'Group 3 (-40 °C)',
	}
	
	return titles[treatment]


def state_to_title(state):
	"""
	Convert state to human readable title
	"""

	titles = {	'fersk' : 'Fresh',
				'Tint' : 'Frozen/thawed'}

	return titles[state]

def process_connected_tissue(images):
	"""
	There are two switches that can set damaged-connected tissue to damaged tissue, and also non-damaged connected tissue to non-damage tissue
	See:
	# set non-damaged connected tissue to non-damaged tissue
	params['set_non_damaged_connected_to_non_damaged'] = True
	# set damaged connected tissue to damaged tissue
	params['set_damaged_connected_to_damaged'] = True 

	"""

	params = get_parameters()

	# check if connected damaged tissue needs to be set to damaged tissue
	if params['set_damaged_connected_to_damaged']:
		# {0 : 'background', 1 : 'damaged',  2 : 'non-damaged', 3: 'damaged connected', 4 : 'non-damaged connected'}
		images[images == 3] = 1
	# check if non-damaged connected needs to be set to non-damaged tissue
	if params['set_non_damaged_connected_to_non_damaged']:
		# {0 : 'background', 1 : 'damaged',  2 : 'non-damaged', 3: 'damaged connected', 4 : 'non-damaged connected'}
		images[images == 4] = 2

	return images

def get_gradcam_methods():
	"""
	Get a list of gradcam methods.

	Parameters
	----------

	Returns
	----------
	gradcam_methods : list
		list of strings referencing gradcam methods
	"""


	# define list of gradcam methods
	# return ['gradcam', 'gradcam_plus_plus', 'scorecam', 'faster_scorecam', 'vanilla_saliency', 'smoothgrad', 'layercam']
	# return ['gradcam', 'gradcam_plus_plus', 'faster_scorecam', 'vanilla_saliency', 'smoothgrad']
	return ['gradcam', 'gradcam_plus_plus', 'scorecam', 'faster_scorecam', 'layercam']

def get_drip_loss():

	paths = get_paths()

	# read drip loss
	drip_loss = pd.read_csv(os.path.join(paths['table_folder'], 'drip_loss', 'drip_loss.csv'), delimiter = ';', decimal = ',')
	# create dictionary of patient to drip loss
	return {f"Torsk {int(row['treatment'])}-{int(row['sample'])} Tint" : row['thawing_loss_percentage'] for _, row in drip_loss.iterrows()}

def cnn_to_title(cnn_type):
	
	titles = {'vgg16' : 'VGG16',
			'resnet50': 'RESNET50',
			'xception': 'XCEPTION',
			'inception_v3': 'INCEPTION V3',
			'resnet_v2': 'INCEPTION V4',
			'densenet121': 'DENSENET 121'}

	return titles[cnn_type]

def gradcam_to_title(gradcam):

	titles = {'gradcam' : 'GradCAM', 
			'gradcam_plus_plus' : 'GradCAM++', 
			'scorecam' : 'ScoreCAM', 
			'faster_scorecam' : 'Faster ScoreCAM',
			'layercam' : 'LayerCAM'}
	
	return titles[gradcam]


def get_supervised_classification(smoothing_size, patient, use_mask = False):

	# read paths data
	paths = get_paths()
	params = get_parameters()

	# read supervised data
	supervised = np.load(os.path.join(paths['supervised_smoothed_folder'], str(smoothing_size), f'{patient}.npy'))
	# process connected tissue in supervised classification
	supervised = process_connected_tissue(supervised)
	# add channel
	supervised = np.expand_dims(supervised, axis = 3)

	if not use_mask:
		return supervised
	else:	
		# get original mri image data
		x = read_dataset_from_group(group_name = 'bg_True_crop_False_gray_False', dataset = patient, hdf5_file = os.path.join(paths['hdf5_folder'], params['hdf5_file']))
		# add channel axes
		x = np.expand_dims(x, axis = 3)

		# mask background of supervised images
		supervised_ma = np.ma.masked_array(data = supervised, mask = (x == 0))

		# adjust class labels to cam 0 = non-damaged, 1 = damaged
		supervised_ma = 1 - (supervised_ma - 1)

		return supervised_ma

def sort_model_subfolders(model_subfolders):
	"""
	Sort model subfolders by parsed out name
	"""
	# add second item to list with parsed name
	model_subfolders = [[x, cnn_to_title(re.search('.*?_(.*)', x).group(1))] for x in model_subfolders]				
	# sort on this new second column
	model_subfolders = sorted(model_subfolders, key = lambda x: x[1])
	# remove second column
	model_subfolders = [x[0] for x in model_subfolders]
	
	return model_subfolders

def sig_to_character(sig_value):

	if sig_value <= 0.001:
		return '***'
	elif sig_value <= 0.01:
		return '**'
	elif sig_value <= 0.05:
		return '*'
	else:
		return ''


def sig_to_text(sig_value):

	if sig_value <= 0.001:
		return 'p < 0.001'
	elif sig_value <= 0.01:
		return 'p < 0.01'
	elif sig_value <= 0.05:
		return 'p < 0.05'
	else:
		return ''


