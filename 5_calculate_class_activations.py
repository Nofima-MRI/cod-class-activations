import os
import numpy as np
import time
import re
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
from itertools import combinations
from scipy.stats import pearsonr, spearmanr, shapiro

# parallel processing
from multiprocessing import cpu_count
from joblib import Parallel
from joblib import delayed

from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.gradcam import GradcamPlusPlus
from tf_keras_vis.layercam import Layercam
from tf_keras_vis.scorecam import Scorecam
from tf_keras_vis.saliency import Saliency
from tf_keras_vis.utils.scores import CategoricalScore
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear

from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K

from functions.helper_functions import set_start, read_directory, create_directory
from functions.project_functions import get_paths, get_parameters, create_patients, get_range_slices, parse_patientname, check_mri_slice_validity, \
										get_mri_image_group_name, process_connected_tissue, get_gradcam_methods, get_drip_loss, sort_model_subfolders
from functions.hdf5_functions import read_dataset_from_group, read_metadata_from_group_dataset
from functions.img_functions import mask_image

def calculate_class_activations(paths, params, limit_patients = None, skip_processed_patients = True):
	"""
	Calculate class activations to indicate what regions of an image are activated by the CNN network

	Parameters
	----------

	"""


	# get a list of all patients to process
	patients = create_patients(treatments = params['limit_treatments'], samples = params['limit_samples'], states = ['Tint'])

	"""
		MODEL LOCATION
	"""	
	# append dataset name to model folder
	model_folder = os.path.join(paths['model_folder'], get_mri_image_group_name())
	# read model subfolders to process
	model_subfolders = [x for x in read_directory(directory = model_folder, subfolders = False) if x != '.DS_Store']
	# get list of gradcam methods
	gradcam_methods = get_gradcam_methods()

	# process each folder
	for model_subfolder in model_subfolders:

		logging.info(f'=== Processing model subfolder {model_subfolder} ===')
	
		# loop over gradcam methods
		for gradcam_method in gradcam_methods:


			logging.info(f"--- Processing gradcam method : {gradcam_method} ---")

			# load cnn model
			model = get_cnn_model(model_folder, model_subfolder)
			replace2linear = ReplaceToLinear()

			"""
				CLASS ACTIVATION INSTANTIATION
			"""
			if gradcam_method == 'gradcam':
				gradcam = Gradcam(model, model_modifier = replace2linear, clone = False)
			elif gradcam_method == 'gradcam_plus_plus':
				gradcam_plus_plus = GradcamPlusPlus(model, model_modifier = replace2linear, clone = False)
			elif gradcam_method in ['scorecam', 'faster_scorecam']:
				scorecam = Scorecam(model)
			elif gradcam_method in ['vanilla_saliency', 'smoothgrad']:
				smoothgrad = Saliency(model, model_modifier = replace2linear, clone=False)
			elif gradcam_method  == 'layercam':
				layercam = Layercam(model, model_modifier = replace2linear, clone = False)
			else:
				logging.error(f"class activation method not implemented: {gradcam_method}")
				exit(1)

			# read patients that have already been processed
			processed_patients = read_directory(os.path.join(paths['class_activation_folder'], get_mri_image_group_name(), model_subfolder, gradcam_method), subfolders = False, return_none_if_not_exists = True)

			# check if folder exists at all, if not, there are no processed patients
			processed_patients = [x[:-4] for x in processed_patients] if processed_patients is not None else []
			
			# loop over patients to calculate class activations per slice
			for patient in patients[:limit_patients]:

				# set starting time
				tic = time.time()

				logging.info(f'Processing patient {patient}')

				# skip patient if already processed
				if skip_processed_patients:
					if patient in processed_patients:
						logging.info(f'Patient {patient} already processed, skipping...')
						continue

				"""
					LOAD FEATURE DATA
				"""
				# get data
				x = read_dataset_from_group(group_name = get_mri_image_group_name(), dataset = patient, hdf5_file = os.path.join(paths['hdf5_folder'], params['hdf5_file']))
				# add channel axes
				x = np.expand_dims(x, axis = 3)
				# rescale image
				x = x * params['rescale_factor']

				# read class label
				y = read_metadata_from_group_dataset(group_name = params['group_original_mri'], dataset = patient, hdf5_file = os.path.join(paths['hdf5_folder'], params['hdf5_file']))['ClassLabel']

				"""
				TESTING
				"""
				y = 1 # since we only do damaged here
				
				# empty array to hold class activations for patient
				class_activations = np.zeros_like(x, dtype = 'float')

				# apply mask if set to True
				# if params['mask_background']:
				# 	x = np.ma.masked_equal(x, 0)

				# loop over each mri slice
				for mri_slice in get_range_slices(mri_data = x):

					# check validity of mri slice
					if check_mri_slice_validity(patient = patient, mri_slice = mri_slice, total_num_slices = x.shape[0]):

						logging.debug(f'Processing slice {mri_slice}')
				
						"""
							Get activations with gradam
						"""
						
						score = CategoricalScore([y])

						if gradcam_method == 'gradcam':
							# Generate heatmap with GradCAM
							cam = gradcam(score = score, seed_input = x[mri_slice])
						elif gradcam_method == 'gradcam_plus_plus':
							# Generate heatmap with GradCAM++
							cam = gradcam_plus_plus(score = score, seed_input = x[mri_slice])
						elif gradcam_method == 'scorecam':
							# Generate heatmap with scorecam
							cam = scorecam(score = score, seed_input = x[mri_slice])
						elif gradcam_method == 'faster_scorecam':
							# Generate heatmap with faster scorecam
							cam = scorecam(score = score, seed_input = x[mri_slice], max_N = 10)
						elif gradcam_method == 'vanilla_saliency':
							cam = smoothgrad(score = score, seed_input = x[mri_slice])
						elif gradcam_method == 'smoothgrad':
							cam = smoothgrad(score = score, seed_input = x[mri_slice], smooth_samples = 20, smooth_noise = 0.20 )
						elif gradcam_method == 'layercam':
							cam = layercam(score = score, seed_input = x[mri_slice])
						
						cam = np.expand_dims(cam, axis = 3)
						
						# fig, axs = plt.subplots(1,2, figsize = (10,10))
						# axs = axs.ravel()

						# axs[0].imshow(x[mri_slice], cmap = 'gray')
						# axs[0].imshow(cam[0], alpha = 0.8)
						# plt.show()
						# exit()
						
						class_activations[mri_slice] = cam[0]
						
				# create subfolder to save numpy file to
				save_subfolder = os.path.join(paths['class_activation_folder'], get_mri_image_group_name(), model_subfolder, gradcam_method)
				# create subfolder
				create_directory(save_subfolder)
				# save class activations as numpy array to disk
				np.save(file = os.path.join(save_subfolder, patient), arr = class_activations)
				# verbose
				logging.info('-- executed in {} seconds'.format(time.time()-tic))
				# clear the graph
				K.clear_session()
				# load model back again
				model = get_cnn_model(model_folder, model_subfolder)
			
def batch_calculate_iou(paths, params, plot_intermediate = False, save_df = True, skip_processed_patients = False):
	"""
		Calculate intersection over union
	"""

	# get a list of all patients to process
	patients = create_patients(treatments = params['limit_treatments'], samples = params['limit_samples'], states = ['Tint'])

	# get model subfolders
	model_subfolders = [x for x in read_directory(directory = os.path.join(paths['model_folder'], get_mri_image_group_name()), subfolders = False) if x != '.DS_Store']

	# process each folder
	for model_subfolder in model_subfolders:

		logging.info(f'=== Processing model subfolder {model_subfolder} ===')
		
		# get available gradcam methods from folder
		gradcam_methods = read_directory(os.path.join(paths['class_activation_folder'], get_mri_image_group_name(), model_subfolder), subfolders=False)

		for gradcam_method in gradcam_methods:

			logging.info(f"--- Processing gradcam method : {gradcam_method} ---")

			# read patients that have class activations
			patients_with_activations = [x[:-4] for x in read_directory(os.path.join(paths['class_activation_folder'], get_mri_image_group_name(), model_subfolder, gradcam_method), subfolders=False)]

			for smoothing_size in params['smoothing_kernels']:

				logging.info(f'--- Processing smoothing size: {smoothing_size} ---')

				# read already processed patients
				processed_patients = read_directory(os.path.join(paths['table_folder'], 'activations_by_threshold', get_mri_image_group_name(), model_subfolder, gradcam_method, str(smoothing_size)), subfolders=False, return_none_if_not_exists=True)
				if processed_patients is None:
					processed_patients = []
				else:
					processed_patients = [x[:-4] for x in processed_patients]

				# dictinary of variables to send to multiprocessing
				p = {
					'patients_with_activations' : patients_with_activations,
					'processed_patients' : processed_patients,
					'skip_processed_patients' : skip_processed_patients,
					'processed_patients' : processed_patients,
					'model_subfolder' : model_subfolder,
					'iou_thresholds' : params['cam_thresholds'],
					'gradcam_method' : gradcam_method,
					'smoothing_size' : smoothing_size,
					'save_df' : save_df,
					'plot_intermediate' : plot_intermediate}

				# parallel processing
				executor = Parallel(n_jobs = cpu_count(), backend = 'multiprocessing')
				# create tasks so we can execute them in parallel
				tasks = (delayed(calculate_iou)(patient = patient, p = p) for patient in patients)
				# start multiprocessing
				executor(tasks)

def calculate_iou(patient, p):

	# check if patient has class activations, if not, then skip
	if patient not in p['patients_with_activations']:
		logging.warning(f'No class activations available for patient {patient}')
		return

	if p['skip_processed_patients']:
		# check if patient has already been processed, if so, then skip
		if patient in p['processed_patients']:
			logging.debug(f'Patient {patient} already processed')
			return

	# check if patient is frozen/thawed
	treatment, sample, state = parse_patientname(patient)

	# dynamically create location for class activations
	f_cam = os.path.join(paths['class_activation_folder'], get_mri_image_group_name(), p['model_subfolder'], p['gradcam_method'], f'{patient}.npy')

	# read activation data
	cam = np.load(f_cam)
	# read supervised data
	supervised = np.load(os.path.join(paths['supervised_smoothed_folder'], str(p['smoothing_size']), f'{patient}.npy'))
	# process connected tissue in supervised classification
	supervised = process_connected_tissue(supervised)
	# add channel
	supervised = np.expand_dims(supervised, axis = 3)
	# get original mri image data
	x = read_dataset_from_group(group_name = 'bg_True_crop_False_gray_False', dataset = patient, hdf5_file = os.path.join(paths['hdf5_folder'], params['hdf5_file']))
	# add channel axes
	x = np.expand_dims(x, axis = 3)

	# mask background of cam images
	cam_ma = np.ma.masked_array(data = cam, mask = (x == 0))
	# mask background of supervised images
	supervised_ma = np.ma.masked_array(data = supervised, mask = (x == 0))

	# mask background of original image
	x = np.ma.masked_equal(x = x, value = 0)

	# adjust class labels to cam 0 = non-damaged, 1 = damaged
	supervised_ma = 1 - (supervised_ma - 1)

	# # convert to bool
	# a = cam_ma.astype('bool')
	# b = supervised_ma.astype('bool')

	# # unrol
	# a = a.reshape(a.shape[0], -1)
	# b = b.reshape(a.shape[0], -1)
	# # calculate intersection
	# intersection = a * b
	# union = a + b
	# iou = np.divide(intersection.sum(axis = 1).reshape(1, -1), union.sum(axis = 1).reshape(1, -1))


	# create empty dataframe
	df = pd.DataFrame()

	# threshold range 
	for threshold in p['iou_thresholds']:

		# logging.debug(f'processing threshold {threshold}')

		# loop over each mri slice
		for mri_slice in get_range_slices(mri_data = x):
		
			# check validity of mri slice
			if check_mri_slice_validity(patient = patient, mri_slice = mri_slice, total_num_slices = x.shape[0]):

				# binarize cam by applying threshold
				a = np.ma.where(cam_ma[mri_slice] >= threshold, 1, 0)

				# assign supervised classification to easy variable
				b = supervised_ma[mri_slice]

				# create boolean array
				a = a.astype('bool')
				b = b.astype('bool')

				# calculate intersection
				intersection = a * b
				# calculate union
				union = a + b
				# calculate intersect over union
				iou = intersection.sum() / float(union.sum())
				# add to dataframe
				df[f'{patient}_{mri_slice}_{threshold}'] = pd.Series({'patient' : patient,
																		'treatment' : treatment,
																		'sample' : sample, 
																		'state' : state, 
																		'mri_slice' : mri_slice, 
																		'threshold' : threshold, 
																		'iou' : iou,
																		'gradcam_method' : p['gradcam_method']})

				# plot if set to True
				if p['plot_intermediate']:

					fig, axs = plt.subplots(1,6, figsize = (15,5))
					axs = axs.ravel()

					# define the colors we want
					plot_colors = ['#250463', '#e34a33']
					# create a custom listed colormap (so we can overwrite the colors of predefined cmaps)
					cmap = colors.ListedColormap(plot_colors)
					
					# original image
					axs[0].imshow(x[mri_slice], cmap = 'gray')
					axs[0].set_title('Original MRI')
					
					# class activations
					axs[1].imshow(cam_ma[mri_slice], interpolation = 'none', cmap = 'jet')
					axs[1].set_title('Activations')
					
					# class activations 
					axs[2].imshow(a, interpolation = 'none', cmap = cmap, vmin = 0, vmax = 1)
					axs[2].set_title(f'binarized activations - {threshold}')
					
					# supervised classification
					axs[3].imshow(b, interpolation = 'none', cmap = cmap, vmin = 0, vmax = 1)
					axs[3].set_title('Supervised Classification')
					
					# intersect
					axs[4].imshow(intersection, interpolation = 'none', vmin = 0, vmax = 1)
					axs[4].set_title(f'intersection - {intersection.sum()}')
					
					# union
					axs[5].imshow(union, interpolation = 'none', vmin = 0, vmax = 1)
					axs[5].set_title(f'union {union.sum()} - iou {round(iou,2)}')
												
					# create plotfolder subfolder
					plot_sub_folder = os.path.join(paths['plot_folder'], 'intermediate', 'iou', get_mri_image_group_name(), p['model_subfolder'], p['gradcam_method'], str(p['smoothing_size']), patient, str(mri_slice))
					# create folder if not exists
					create_directory(plot_sub_folder)

					# crop white space
					fig.set_tight_layout(True)
					# save the figure
					fig.savefig(os.path.join(plot_sub_folder, f'{mri_slice}_{threshold}.png'))

					# close the figure environment
					plt.close()

	if p['save_df']:
		# transpose dataframe
		df = df.T
		# save dataframe to file
		save_folder = os.path.join(paths['table_folder'], 'activations_by_threshold', get_mri_image_group_name(), p['model_subfolder'], p['gradcam_method'], str(p['smoothing_size']))
		# create that folder
		create_directory(save_folder)
		# save dataframe to folder
		df.to_csv(os.path.join(save_folder, f'{patient}.csv'))

def batch_process_get_iou_table(params, paths):

	# list of patients
	patients = create_patients(treatments = params['limit_treatments'], samples = params['limit_samples'], states = ['Tint'])	
	# read available models
	model_subfolders = [x for x in read_directory(directory = os.path.join(paths['table_folder'], 'activations_by_threshold', get_mri_image_group_name()), subfolders = False)]

	# overwrite smoothing kernsl
	# params['smoothing_kernels'] = [4]

	# combinations between smoothing kernel and cam threshold
	batch_combinations = [(x, y) for x in params['smoothing_kernels'] for y in params['cam_thresholds']]
	
	# parallel processing
	executor = Parallel(n_jobs = cpu_count(), backend = 'multiprocessing')
	# create tasks so we can execute them in parallel
	tasks = (delayed(get_iou_table)(smoothing_size = com[0], threshold = com[1], model_subfolders = model_subfolders, patients = patients) for com in batch_combinations)
	# start multiprocessing
	executor(tasks)

def get_iou_table(smoothing_size, threshold, model_subfolders, patients):

	logging.info(f'Processing smoothing : {smoothing_size}, threshold : {threshold}')

	paths = get_paths()

	# create empty dataframe (to be populated later)
	df = pd.DataFrame(index = model_subfolders, columns = [f'{x}-{y}' for x in range(1,4) for y in get_gradcam_methods()])
	# another dataframe with only values (no text input)
	df_values = df.copy()
	
	for row_index, row in df.iterrows():

		for col_index, col in row.items():

			# parse out values from column index
			treatment, gradcam_method = col_index.split('-')
			# assign row_index to different variable to make it consistent
			model_subfolder = row_index

			# logging.debug(f'Processing {model_subfolder}, {col_index}')

			# create data folder
			data_folder = os.path.join(paths['table_folder'], 'activations_by_threshold', get_mri_image_group_name(), model_subfolder, gradcam_method, str(smoothing_size))
			# empty datframe to read data to
			cam_df = pd.DataFrame()

			# read patients for which we have data
			available_patients = [x[:-4] for x in read_directory(data_folder, subfolders = False)]
			
			# read in data
			for patient in patients:
				# check if there is patient data
				if patient in available_patients:
					# read in data
					df_add = pd.read_csv(os.path.join(data_folder, f'{patient}.csv'))
					# add to dataframe
					cam_df = pd.concat([cam_df, df_add], ignore_index=True)
				else:
					logging.warning(f'No patient data for: {patient}')

			# filter dataframe
			cam_df_filtered = cam_df.query(f"treatment == {treatment} & threshold == {threshold} ")
			# calculate average iou
			avg_iou = cam_df_filtered['iou'].mean()
			std_iou = cam_df_filtered['iou'].std()

			# add to cell
			df.at[row_index, col_index] = f'{round(avg_iou,2)} (±{round(std_iou,2)})'
			df_values.at[row_index, col_index] = avg_iou

	# save folder
	save_folder = os.path.join(paths['table_folder'], 'avg_iou_by_threshold', 'text')
	# create that folder
	create_directory(save_folder)
	# save dataframe to folder
	df.to_csv(os.path.join(save_folder, f'{smoothing_size}_{threshold}.csv'))
	df.to_excel(os.path.join(save_folder, f'{smoothing_size}_{threshold}.xls'))

	# save folder
	save_folder = os.path.join(paths['table_folder'], 'avg_iou_by_threshold', 'values')
	# create that folder
	create_directory(save_folder)
	# save dataframe to folder
	df_values.to_csv(os.path.join(save_folder, f'{smoothing_size}_{threshold}.csv'))

def get_best_smoothing_kernel_size(paths, params):

	# read folder with data
	F = read_directory(os.path.join(paths['table_folder'], 'avg_iou_by_threshold', 'values'))

	# empty dictionary to hold data
	data = {x : [] for x in params['smoothing_kernels']}


	for f in F:

		# read data
		D = pd.read_csv(f, index_col = 0)

		total_avg = D.stack().mean()
		
		# extract kernel from file name
		kernel_size = int(f.split(os.sep)[-1].split('_')[0])
		
		# add to data
		data[kernel_size].append(total_avg)

	for k, v in data.items():
		avg = np.array(v).mean()
		print(k, avg)
	"""
	1 0.3178176027245788
	4 0.3311850853254201 <- best
	8 0.328072495843602
	12 0.32663163439003273
	16 0.3279758664510088
	20 0.3294163480190977
	24 0.33128481161999973
	28 0.33312642611232945
	32 0.3348977187784672

	"""

def batch_get_damaged_by_threshold(paths, params):

	# patients to process
	patients = create_patients(treatments = params['limit_treatments'], samples = params['limit_samples'], states = ['Tint'])	

	# append dataset name to model folder
	model_folder = os.path.join(paths['model_folder'], get_mri_image_group_name())
	# read model subfolders to process
	model_subfolders = [x for x in read_directory(directory = model_folder, subfolders = False) if x != '.DS_Store']

	# plot activations for each model subfolder
	for model_subfolder in model_subfolders:

		logging.info(f'=== Processing model subfolder {model_subfolder} ===')

		# check if activations are constructed for gradcam method
		activations_for_gradcam_method_available = read_directory(os.path.join(paths['class_activation_folder'], get_mri_image_group_name(), model_subfolder), subfolders = False)
		# check if gradcam method has alread been processed with damaged by threshold values
		processed_gradcam_methods = read_directory(os.path.join(paths['table_folder'], 'damaged_by_treshold', get_mri_image_group_name(), model_subfolder), subfolders = False, return_none_if_not_exists = True)
		# check for None return
		if processed_gradcam_methods is None:
			processed_gradcam_methods = []

		for gradcam_method in get_gradcam_methods():


			logging.info(f'--- Processing gradcam method: {gradcam_method} ---')

			# check if gradcam method has been processed for activations already
			if gradcam_method not in activations_for_gradcam_method_available:
				logging.warning(f'No class activations found for gradcam method: {gradcam_method}')
				continue

			# check if gradcam method has already been processed for damage by threshold
			# if gradcam_method in processed_gradcam_methods:
			# 	logging.debug(f'Gradcam method {gradcam_method} already processed, skipping....')
			# 	continue

			# empty dataframe to store data
			df = pd.DataFrame()

			# read class activations file location
			F = read_directory(os.path.join(paths['class_activation_folder'], get_mri_image_group_name(), model_subfolder, gradcam_method))

			# parallel processing
			executor = Parallel(n_jobs = cpu_count(), backend = 'multiprocessing')
			# create tasks so we can execute them in parallel
			tasks = (delayed(get_damaged_by_threshold)(file = f, allow_patients = patients, gradcam_method = gradcam_method) for f in F)
			# start multiprocessing
			for df_add in executor(tasks):

				df = pd.concat([df, df_add], axis=1)
				
			# transpose
			df = df.T
			# save dataframe to file
			save_folder = os.path.join(paths['table_folder'], 'damaged_by_treshold', get_mri_image_group_name(), model_subfolder, gradcam_method)
			# create that folder
			create_directory(save_folder)
			# save dataframe to folder
			df.to_csv(os.path.join(save_folder, 'data.csv'))

def get_damaged_by_threshold(file, allow_patients, gradcam_method):

	params = get_parameters()

	# read in activations
	cam = np.load(file)

	# extract patient name
	patient = file.split(os.sep)[-1][:-4]

	# empty dataframe to store data
	df = pd.DataFrame()

	for threshold in params['cam_thresholds']:

		# check if patient is tint
		if patient in allow_patients:

			# logging.info(f'Processing patient : {patient}')

			# get mask background
			# dynamically create hdf5 file
			hdf5_file = os.path.join(paths['hdf5_folder'], params['hdf5_file'])
			data = read_dataset_from_group(group_name = 'bg_True_crop_False_gray_False', dataset = patient, hdf5_file = hdf5_file)

			# apply mask to remove background
			cam_ma = mask_image(img = cam, segmented_img = data, mask_value = 0, fill_value = None)

			damaged = np.array(np.sum(np.count_nonzero(cam_ma >= threshold, axis = 1), axis = 1).squeeze())
			non_damaged = np.array(np.sum(np.count_nonzero(cam_ma < threshold, axis = 1), axis = 1).squeeze())
			rel_damaged = damaged / (damaged + non_damaged)

			cam_avg = np.mean(np.ma.masked_where(cam_ma < threshold, cam_ma).reshape(cam_ma.shape[0], -1), axis = 1)

			# process each slice	
			for mri_slice in get_range_slices(data):

				# check validity of mri slice
				if check_mri_slice_validity(patient = patient, mri_slice = mri_slice, total_num_slices = data.shape[0]):

					# check value is numeric
					# if isinstance(cam_avg[mri_slice], np.floating):

						# if non_damaged[mri_slice] == 0:
						# 	continue

					# create index
					df_index = f'{gradcam_method}_{patient}_{mri_slice}_{threshold}'
					
					df_series = pd.Series({	'patient' : patient,
										'mri_slice' : mri_slice,
										'damaged' : damaged[mri_slice],
										'non_damaged' : non_damaged[mri_slice],
										'rel_damaged' : rel_damaged[mri_slice],
										'cam_avg' : cam_avg[mri_slice],
										'threshold' : threshold,
										'gradcam_method' : gradcam_method
										})
					
					df[df_index] = df_series
						
	return df

def calculate_correlation_with_drip_loss(paths):

	# get drip loss per patient
	patient_drip_loss = get_drip_loss()

	# read model subfolders to process
	model_subfolders = [x for x in read_directory(directory = os.path.join(paths['table_folder'], 'damaged_by_treshold', get_mri_image_group_name()), subfolders = False) if x != '.DS_Store']

	# empty dataframe to keep plot data
	df = pd.DataFrame()
					
	# plot activations for each model subfolder
	for model_subfolder in model_subfolders:

		logging.info(f'=== Processing model subfolder {model_subfolder} ===')
		
		# read available gradcam methods
		gradcam_methods = [x for x in read_directory(directory = os.path.join(paths['table_folder'], 'damaged_by_treshold', get_mri_image_group_name(), model_subfolder), subfolders = False) if x != '.DS_Store']

		# process available gradcam methods
		for gradcam_method in gradcam_methods:

			logging.info(f"--- Processing gradcam method : {gradcam_method} ---")
			
			# read in damaged data
			tissue_data = pd.read_csv(os.path.join(paths['table_folder'], 'damaged_by_treshold', get_mri_image_group_name(), model_subfolder, gradcam_method, 'data.csv'), index_col = 0)

			# empty dataframe for temp plot data
			temp_plot_data = pd.DataFrame()

			# get average damage per patient/sample
			for (patient, threshold), group_data in tissue_data.groupby(['patient', 'threshold']):
				
				temp_plot_data[f'{patient}_{threshold}'] = pd.Series({	'damaged' : group_data['rel_damaged'].astype('float').mean(),
																		'damaged_std' : group_data['rel_damaged'].astype('float').std(),
																		'patient' : patient,
																		'threshold' : threshold,
																		'drip_loss' : patient_drip_loss[patient]})
			
			# calculate correlation per threshold
			logging.debug('Calculating correlation')
			for threshold, group_data in temp_plot_data.T.groupby('threshold'):



				# calculate if data comes from normal distribution
				s_statistic, s_pvalue = shapiro(group_data['damaged'])
				# calculate correlation pearson correlation
				pearson_r, pearson_p = pearsonr(x = group_data['damaged'], y = group_data['drip_loss'])
				# calculate spearman correlation
				spearman_r, spearman_p = spearmanr(group_data['damaged'], group_data['drip_loss'])

				# add to plot data
				df[f'{model_subfolder}_{gradcam_method}_{threshold}'] = pd.Series({	'model_subfolder' : model_subfolder,
																					'gradcam_method' : gradcam_method,
																					'threshold' : threshold,
																					'pearson_r' : pearson_r,
																					'pearson_p' : pearson_p,
																					'spearman_r' : spearman_r,
																					'spearman_p' : spearman_p,
																					's_statistic' : s_statistic,
																					's_pvalue' : s_pvalue,
																					'damaged' : group_data['damaged'].mean(),
																					'damaged_std' : group_data['damaged_std'].std()})


	# tranpose
	df = df.T
	# save dataframe to file
	save_folder = os.path.join(paths['table_folder'], f'correlation_with_driploss', get_mri_image_group_name())
	# create that folder
	create_directory(save_folder)
	# save dataframe to folder
	df.to_csv(os.path.join(save_folder, 'data.csv'))		
			
def iou_by_cam(paths, params):


	# patients to process
	patients = create_patients(treatments = params['limit_treatments'], samples = params['limit_samples'], states = ['Tint'])	

	# append dataset name to model folder
	model_folder = os.path.join(paths['model_folder'], get_mri_image_group_name())
	# read model subfolders to process
	model_subfolders = [x for x in read_directory(directory = model_folder, subfolders = False) if x != '.DS_Store']

	# empty dataframe to collect all data
	df = pd.DataFrame()

	for threshold in params['cam_thresholds']:

		logging.info(f'Processing threshold : {threshold}')

		for patient in patients:

			logging.info(f'processing patients : {patient}')

			for model_subfolder in model_subfolders:

				# get original mri image data
				x = read_dataset_from_group(group_name = 'bg_True_crop_False_gray_False', dataset = patient, hdf5_file = os.path.join(paths['hdf5_folder'], params['hdf5_file']))
				# add channel axes
				x = np.expand_dims(x, axis = 3)
				# create array first
				array_data = {}

				# loop over combinations of gradcam methods
				for gradcam_method in get_gradcam_methods():

					# dynamically create location for class activations
					f_cam = os.path.join(paths['class_activation_folder'], get_mri_image_group_name(), model_subfolder, gradcam_method, f'{patient}.npy')
					# read activation data
					a = np.load(f_cam)
					# mask background of cam images
					a = np.ma.masked_array(data = a, mask = (x == 0))	
					# binarize cam by applying threshold
					a = np.ma.where(a >= threshold, 1, 0)
					# create boolean array
					a = a.astype('bool')
					# unrol
					a = a.reshape(a.shape[0], -1)
					# add to dictionary
					array_data[gradcam_method] = a

				# cam combinations
				combos = [x for x in combinations(get_gradcam_methods(),2)]
				# empty array to hold data
				data = np.zeros(shape = (54, len(combos)))

				# loop over combinations of gradcam methods
				for i, (gradcam_method_1, gradcam_method_2) in enumerate(combos):

					# logging.info(f'Processing gradcam methods: {gradcam_method_1} with {gradcam_method_2} ')

					a = array_data[gradcam_method_1]
					b = array_data[gradcam_method_2]

					# calculate intersection
					intersection = a * b
					union = a + b

					iou = np.divide(intersection.sum(axis = 1).reshape(1, -1), union.sum(axis = 1).reshape(1, -1))

					# add to array
					data[:,i] = iou.reshape(-1)


				# get averages over rows
				row_averages = np.mean(data, axis = 1)

				# loop over each mri slice
				for mri_slice in get_range_slices(mri_data = x):
				
					# check validity of mri slice
					if check_mri_slice_validity(patient = patient, mri_slice = mri_slice, total_num_slices = x.shape[0]):

						# add data to dataframe
						df[f'{threshold}_{model_subfolder}_{patient}_{mri_slice}'] = pd.Series({
							'threshold' : threshold,
							'model_subfolder' : model_subfolder,
							'patient' : patient,
							'mri_slice' : mri_slice,
							'iou' : row_averages[mri_slice]
						})
	
	# tranpose
	df = df.T
	# save dataframe to file
	save_folder = os.path.join(paths['table_folder'], f'iou_by_cam', get_mri_image_group_name())
	# create that folder
	create_directory(save_folder)
	# save dataframe to folder
	df.to_csv(os.path.join(save_folder, 'data.csv'))		
			
def iou_by_model(paths, params):

	# patients to process
	patients = create_patients(treatments = params['limit_treatments'], samples = params['limit_samples'], states = ['Tint'])	

	# append dataset name to model folder
	model_folder = os.path.join(paths['model_folder'], get_mri_image_group_name())
	# read model subfolders to process
	model_subfolders = [x for x in read_directory(directory = model_folder, subfolders = False) if x != '.DS_Store']

	# empty dataframe to collect all data
	df = pd.DataFrame()

	for threshold in params['cam_thresholds']:

		logging.info(f'Processing threshold : {threshold}')

		for patient in patients:

			logging.info(f'processing patients : {patient}')

			# loop over combinations of gradcam methods
			for gradcam_method in get_gradcam_methods():

				# get original mri image data
				x = read_dataset_from_group(group_name = 'bg_True_crop_False_gray_False', dataset = patient, hdf5_file = os.path.join(paths['hdf5_folder'], params['hdf5_file']))
				# add channel axes
				x = np.expand_dims(x, axis = 3)
				# create array first
				array_data = {}

				for model_subfolder in model_subfolders:

					# dynamically create location for class activations
					f_cam = os.path.join(paths['class_activation_folder'], get_mri_image_group_name(), model_subfolder, gradcam_method, f'{patient}.npy')
					# read activation data
					a = np.load(f_cam)
					# mask background of cam images
					a = np.ma.masked_array(data = a, mask = (x == 0))	
					# binarize cam by applying threshold
					a = np.ma.where(a >= threshold, 1, 0)
					# create boolean array
					a = a.astype('bool')
					# unrol
					a = a.reshape(a.shape[0], -1)
					# add to dictionary
					array_data[model_subfolder] = a

				# cam combinations
				combos = [x for x in combinations(model_subfolders,2)]
				# empty array to hold data
				data = np.zeros(shape = (54, len(combos)))

				# loop over combinations of gradcam methods
				for i, (model_subfolder_1, model_subfolder_2) in enumerate(combos):

					# logging.info(f'Processing gradcam methods: {gradcam_method_1} with {gradcam_method_2} ')

					a = array_data[model_subfolder_1]
					b = array_data[model_subfolder_2]

					# calculate intersection
					intersection = a * b
					union = a + b

					iou = np.divide(intersection.sum(axis = 1).reshape(1, -1), union.sum(axis = 1).reshape(1, -1))

					# add to array
					data[:,i] = iou.reshape(-1)

				# get averages over rows
				row_averages = np.mean(data, axis = 1)

				# loop over each mri slice
				for mri_slice in get_range_slices(mri_data = x):
				
					# check validity of mri slice
					if check_mri_slice_validity(patient = patient, mri_slice = mri_slice, total_num_slices = x.shape[0]):

						# add data to dataframe
						df[f'{threshold}_{gradcam_method}_{patient}_{mri_slice}'] = pd.Series({
							'threshold' : threshold,
							'gradcam_method' : gradcam_method,
							'patient' : patient,
							'mri_slice' : mri_slice,
							'iou' : row_averages[mri_slice]
						})
	# tranpose
	df = df.T
	# save dataframe to file
	save_folder = os.path.join(paths['table_folder'], f'iou_by_model', get_mri_image_group_name())
	# create that folder
	create_directory(save_folder)
	# save dataframe to folder
	df.to_csv(os.path.join(save_folder, 'data.csv'))		



def calculate_drip_loss_correlation_with_iou(paths, params):

	# get drip loss per patient
	patient_drip_loss = get_drip_loss()
	# read available models
	model_subfolders = [x for x in read_directory(directory = os.path.join(paths['table_folder'], 'damaged_by_treshold', get_mri_image_group_name()), subfolders = False) if x != '.DS_Store']
	# sort models
	model_subfolders = sort_model_subfolders(model_subfolders)

	# empty dataframe to store correlation per model and per gradcam method
	df_cor = pd.DataFrame()

	# process each model
	for model_subfolder in model_subfolders:

		logging.info(f'=== Processing model subfolder {model_subfolder} ===')
		
		# read available gradcam methods
		gradcam_methods = [x for x in read_directory(directory = os.path.join(paths['table_folder'], 'damaged_by_treshold', get_mri_image_group_name(), model_subfolder), subfolders = False) if x != '.DS_Store']

		# process available gradcam methods
		for gradcam_method in gradcam_methods:

			# empty dataframe to keep plot data
			plot_data = pd.DataFrame()
		
			logging.info(f"--- Processing gradcam method : {gradcam_method} ---")
			
			# read in damaged data
			tissue_data = pd.read_csv(os.path.join(paths['table_folder'], 'damaged_by_treshold', get_mri_image_group_name(), model_subfolder, gradcam_method, 'data.csv'), index_col = 0)

			# get average damage per patient/sample
			for (patient, threshold), group_data in tissue_data.groupby(['patient', 'threshold']):

				# parse out treatment, sample, and stat e from patient name
				treatment, sample, state = parse_patientname(patient_name = patient)

				# check if threshold should be processed
				if threshold in params['cam_thresholds']:
							
					plot_data[f'{patient}_{threshold}'] = pd.Series({'rel_damaged' : group_data['rel_damaged'].astype('float').mean(),
																	'patient' : patient,
																	'threshold' : threshold,
																	'treatment' : treatment,
																	'sample' : sample,
																	'state' : state,
																	'treatment' : treatment,
																	'drip_loss' : patient_drip_loss[patient]})

			# tranpose dataframe
			plot_data = plot_data.T
			# add color column
			# cast to float
			plot_data['threshold'] == plot_data['threshold'].astype('float') 
			plot_data['rel_damaged'] == plot_data['rel_damaged'].astype('float') * 100
			plot_data['drip_loss'] == plot_data['drip_loss'].astype('float')

			print(plot_data)
			exit()

			# process each plot by threshold
			for threshold in params['cam_thresholds']:

				# filter data by threshold
				data = plot_data.query(f"threshold == {threshold}")
				# set certains columns to float
				data = data.astype({'rel_damaged': float,'drip_loss': float})

				# calculate correlation pearson correlation
				pearson_r, pearson_p = pearsonr(x = data['rel_damaged'], y = data['drip_loss'])
				# calculate spearman correlation
				spearman_r, spearman_p = spearmanr(data['rel_damaged'], data['drip_loss'])
			
				# add to table
				df_cor[f'{model_subfolder}_{gradcam_method}_{threshold}'] = pd.Series({'model_subfolder' : model_subfolder,
																						'gradcam_method' : gradcam_method,
																						'threshold' : threshold,
																						'pearson_r' : pearson_r,
																						'pearson_p' : pearson_p,
																						'spearman_r' : spearman_r,
																						'spearman_p' : spearman_p})
			
	df_cor = df_cor.T
	# save dataframe to file
	save_folder = os.path.join(paths['table_folder'], f'drip_loss_correlation_with_iou', get_mri_image_group_name())
	# create that folder
	create_directory(save_folder)
	# save dataframe to folder
	df_cor.to_csv(os.path.join(save_folder, 'data.csv'))		


def get_cnn_model(model_folder, model_subfolder):

	# get project parameters
	params = get_parameters()

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
		
		logging.info(f'Loading checkpoint model from : {checkpoint_model_file}')
		# load checkpoint model
		model = load_model(checkpoint_model_file)

	return model


if __name__ == "__main__":

	tic, process, logging = set_start()

	# get all folder paths
	paths = get_paths()
	# get project parameters
	params = get_parameters()

	# # 1) calculate clas activations with different cam methods
	# calculate_class_activations(paths = paths, params = params)

	# # 2) calculate intersect over union with supervised classification
	# batch_calculate_iou(paths = paths, params = params)

	# # 3) get IOU table
	# batch_process_get_iou_table(paths = paths, params = params)

	# # get best smoothing kernel size
	# get_best_smoothing_kernel_size(paths = paths, params = params)

	# # 4) get damaged by threshold
	# batch_get_damaged_by_threshold(paths = paths, params = params)

	# # 5) calculate correlation with drip loss
	# calculate_correlation_with_drip_loss(paths = paths)

	# # 5) iou comparing CAM methods with each other
	# iou_by_cam(paths = paths, params = params)

	# # 6) iou comparing CAM from different models with each other
	iou_by_model(paths = paths, params = params)

	# 7) drip loss correlation vs iou with supervised classification
	# calculate_drip_loss_correlation_with_iou(paths = paths, params = params)
