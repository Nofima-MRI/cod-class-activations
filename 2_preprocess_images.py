import os
import numpy as np
import matplotlib.pyplot as plt
from cv2 import cv2

from itertools import cycle
from scipy import ndimage

# parallel processing
from multiprocessing import cpu_count
from joblib import Parallel
from joblib import delayed

from functions.helper_functions import set_start, set_end, create_directory
from functions.hdf5_functions import read_dataset_from_group, read_metadata_from_group_dataset, save_data_to_group_hdf5
from functions.img_functions import perform_knn_segmentation, mask_image, change_img_contrast, perform_piecewise_linear_transformation, perform_histogram_equalization, smooth_image_most_occuring_value,\
									perform_adaptive_histogram_equalization
from functions.project_functions import get_mri_image_group_name, get_parameters, get_paths, create_patients, get_mri_image_group_name

def remove_background(data, patient, plot_intermediate_slices = False, save_plots = False):
	"""
	Remove background from MRI images

	Parameters
	--------------
	data : np.array()
		numpy array with MRI images, for example of shape (54, 256, 256)
	patient : string
		name of patient to process, for example Torsk 1-2 Tint (this can be necessary for patient specific parameters)
	"""

	# new numpy array to hold segmented data
	data_segmented = np.empty_like(data, dtype = 'int16')

	# process each slice
	for i in range(data.shape[0]):

		if plot_intermediate_slices:
			ind_cycle = cycle(range(10))
			fig, axs = plt.subplots(1,8, figsize = (20,5))
			axs = axs.ravel()
		
		# original MRI
		img = data[i]
		if plot_intermediate_slices:
			plt_index = next(ind_cycle)
			axs[plt_index].imshow(img, cmap = 'gray')
			axs[plt_index].set_title('Original MRI')

		# change grayscale
		img = change_img_contrast(img, phi = 10, theta = 1)
		if plot_intermediate_slices:
			plt_index = next(ind_cycle)
			axs[plt_index].imshow(img, cmap = 'gray')
			axs[plt_index].set_title('Changed gray scale')


		# convert to 8 bit
		if patient not in ['Torsk 1-4 fersk']:
			img = np.array(img, dtype = 'uint8')

			if plot_intermediate_slices:	
				plt_index = next(ind_cycle)
				axs[plt_index].imshow(img, cmap = 'gray')
				axs[plt_index].set_title('Convert to 8 bit')

		# max filter
		img = ndimage.maximum_filter(img, size = 7)
		if plot_intermediate_slices:
			plt_index = next(ind_cycle)
			axs[plt_index].imshow(img, cmap = 'gray')
			axs[plt_index].set_title('Max filter')

		# erosion
		img = cv2.erode(img, None, iterations = 4)
		if plot_intermediate_slices:
			plt_index = next(ind_cycle)
			axs[plt_index].imshow(img, cmap = 'gray')
			axs[plt_index].set_title('Erosion')


		# gaussian filter
		img = cv2.GaussianBlur(img, (11, 11), 0)
		if plot_intermediate_slices:
			plt_index = next(ind_cycle)
			axs[plt_index].imshow(img, cmap = 'gray')
			axs[plt_index].set_title('Gaussian Blur')

		# knn bg remove
		segmented_img = perform_knn_segmentation(n_clusters = 2, img = img)
		img = mask_image(img = data[i], segmented_img = segmented_img, mask_value = segmented_img[0][0], fill_value = 0)
		if plot_intermediate_slices:
			plt_index = next(ind_cycle)
			axs[plt_index].imshow(img, cmap = 'gray')
			axs[plt_index].set_title('KNN BG remove')

			# show plot and continue to next one after closing plot
			plt.show()
			plt.close()

		# add masked image to data_segmented, where we store each slice
		data_segmented[i] = img

	"""
		Save plot to file
	"""
	if save_plots:

		# mask plots
		data_segmented = np.ma.masked_equal(data_segmented, value = 0)

		fig, axs = plt.subplots(8,7, figsize = (15,15))
		axs = axs.ravel()

		for i in range(data_segmented.shape[0]):
			axs[i].imshow(data_segmented[i], cmap = 'gray')
			axs[i].set_title('Slice {}'.format(i))
			axs[i].set_xticks([])
			axs[i].set_yticks([])
			# axs[i].axis('off')
			
		# image plot folder
		image_plot_folder = os.path.join(paths['plot_folder'], 'intermediate' , 'background_removal')
		# create folder to store image to
		create_directory(image_plot_folder)

		# save the figure
		fig.savefig(os.path.join(image_plot_folder, f'{patient}.png'), dpi = 300)
		
		# close the plot environment
		plt.close()

	# return data
	return data_segmented

def create_cropped_images(data, meta_data, params):
	"""
	Crop image from static or dynamic midpoint. Static midpoint is cropping from the pixel in the center. Dynamical midpoints are given by the function 3_annotate_midpoint and have a more precise middle point.
	"""

	# get dynamic midpoint
	if params['dynamic_midpoint']:
		# read dynamic mid points (meaning the middle of the tissue)
		mid_x = meta_data.get('midpoint_y')
		mid_y = meta_data.get('midpoint_x')

		# check if None
		if mid_x is None or mid_y is None:
			logging.error('Dynamic midpoints for x and y coordinates do not exists. Run 3_annotate_midpoint first')
			exit()
		else:
			# create integer
			mid_x = int(mid_x)
			mid_y = int(mid_y)
	else:
		# dynamic midpoint not True then simpy take the middle
		mid_x = data.shape[1] // 2
		mid_y = data.shape[2] // 2

	# get crop dimensions
	crop_x, crop_y = params['crop_dimensions']
	# new array for cropped images
	cropped_images = np.empty((data.shape[0], crop_x, crop_y))
		
	# loop over each slice and cropp image
	for i in range(data.shape[0]):
		
		# crop image slice
		cropped_images[i] = data[i][mid_x - (crop_x // 2) : mid_x + (crop_x // 2), mid_y - (crop_y // 2) : mid_y + (crop_y // 2)]

	return cropped_images

def perform_adjust_grayscale(data, patient, params, create_plots = False):

	# empty array to holdnew data
	adjusted_data = np.empty_like(data, dtype = 'float')
	# empty dictionary to hold new meta_data
	new_meta_data = {}

	# parallel processing
	executor = Parallel(n_jobs = cpu_count(), backend = 'multiprocessing')

	# check what type of grayscale adjustment to execute
	if params['gray_scale_method'] == 'contrast':
		logging.debug('Performing contrast stretching')

		# convert data to float
		data = data.astype('float')

		# create tasks so we can execute them in parallel
		tasks = (delayed(perform_contrast_stretching)(i = i, data = data, params = params) for i in range(data.shape[0]))

		for new_data, i, r1, r2 in executor(tasks):

			# add new data that contains contrast stretched image to adjusted data array
			adjusted_data[i] = new_data

			# add meta data
			new_meta_data[i] = {'r1' : r1, 'r2' : r2}
	
	elif params['gray_scale_method'] == 'histogram':

		logging.debug('Performing histogram equalization')

		# create tasks so we can execute them in parallel
		tasks = (delayed(perform_histogram_equalization)(img = np.ma.masked_equal(data[i], 0), i = i) for i in range(data.shape[0]))

		for new_data, i in executor(tasks):

			# add new data that contains contrast stretched image to adjusted data array
			adjusted_data[i] = new_data
	
	elif params['gray_scale_method'] == 'adap_histogram':

		logging.debug('Performing adaptive histogram equalizatin')

		# create tasks so we can execute them in parallel
		tasks = (delayed(perform_adaptive_histogram_equalization)(img = data[i], clip_limit = 0.03, i = i) for i in range(data.shape[0]))

		for new_data, i in executor(tasks):

			# rescale data to 12 bit again
			new_data *= 2**12

			# add new data that contains contrast stretched image to adjusted data array
			adjusted_data[i] = new_data
	


	else:
		logging.error(f"Grayscale adjustment method {params['gray_scale_method']} not implemented")
		exit(1)

	"""
		PLOTTING OF INTERMEDIATE SLICES
	"""

	# plot original and adjusted image including histogram
	if create_plots:
		# plot each slice
		for i in range(data.shape[0]):

			logging.info(f'Processing plot {patient}, slice : {i}')

			# create plot
			fig, axs = plt.subplots(2, 2, figsize = (10,10))
			axs = axs.ravel()

			# original data
			axs[0].imshow(data[i], cmap = 'gray')
			# original histogram
			axs[1].hist(x = data[i].reshape(-1), bins = 100, density = False, range = (0,4096), log = False)
			# contrast adjusted image
			axs[2].imshow(adjusted_data[i], cmap = 'gray')
			# contrast adjusted histogram
			axs[3].hist(x = adjusted_data[i].reshape(-1), bins = 100, density = False, range = (0,4096), log = False)
			

			# create title
			axs[0].set_title(f'{patient} - Original')
			axs[2].set_title(f"{patient} - {params['gray_scale_method']} Adjusted")
			if params['gray_scale_method'] == 'contrast':
				axs[3].set_title(f"r1={int(new_meta_data[i]['r1'])}, r2={int(new_meta_data[i]['r2'])}")

			# create plotfolder subfolder
			plot_sub_folder = os.path.join(paths['plot_folder'], 'intermediate', 'gray_scale_adjustment', params['gray_scale_method'], patient)
			# create folder if not exists
			create_directory(plot_sub_folder)

			# crop white space
			fig.set_tight_layout(True)
			# save the figure
			fig.savefig(os.path.join(plot_sub_folder, f'{i}.png'))

			# close the figure environment
			plt.close()

	return adjusted_data

def perform_contrast_stretching(i, data, params, verbose = True):
	"""
	Code adjusted from:
	https://www.geeksforgeeks.org/python-intensity-transformation-operations-on-images/

	"""

	logging.debug(f'Processing slice: {i}/{data.shape[0]}')

	# get correct slice from data
	x = data[i].copy()
	# set zero to nan
	x[x==0] = np.nan

	"""
		Constrast Stretching
	"""

	# adjustment parameters
	r1 = np.nanpercentile(x, q = params['contrast_percentile_lower_bound'])
	s1 = 0 # min value
	r2 = np.nanpercentile(x, q = params['contrast_percentile_upper_bound'])
	s2 = 2**12-1 # max value (12bit)

	if verbose:
		logging.debug(f'r1 : {r1}, r2 : {r2}')

	# vectorize piecewise linear transformation function so it can be applied to all pixels in an array
	lin_transf_vec = np.vectorize(perform_piecewise_linear_transformation)
		
	# get contrast stretched image
	new_data = lin_transf_vec(data[i], r1, s1, r2, s2)

	return new_data, i, r1, r2

def perform_preprocessing(paths, params):
	"""
	Start preprocessing of the data. This can include, removal background, cropping of the images, grayscale adjustment

	Note that the parameters to control preprocessing steps are given in params, most notably the following:
	params['use_no_bg_images']
	params['use_cropped_images']
	params['adjust_gray_scale']
	params['gray_scale_method']

	"""

	# dynamically create hdf5 file
	hdf5_file = os.path.join(paths['hdf5_folder'], params['hdf5_file'])

	# create list of patient names for filtering
	patients = create_patients(treatments = params['limit_treatments'], samples = params['limit_samples'], states = params['limit_states'])

	# read data from each dataset and plot mri data
	for i, patient in enumerate(patients):

		logging.info(f'Processing patient : {patient} {i + 1}/{len(patients)}')

		# read data from group	
		data = read_dataset_from_group(group_name = params['group_original_mri'], dataset = patient, hdf5_file = hdf5_file)
		# read meta data
		meta_data = read_metadata_from_group_dataset(group_name = params['group_original_mri'], dataset = patient, hdf5_file = hdf5_file)

		"""
			BACKGROUND REMOVAL
		"""
		if params['use_no_bg_images']:

			logging.debug('Start removing background from images')
			# remove background image
			data = remove_background(data = data, patient = patient)

		"""
			GRAYSCALE ADJUSTMENT
		"""
		if params['adjust_gray_scale']:
			logging.info('Start gray scale adjustment')

			# adjust gray scale
			data = perform_adjust_grayscale(data = data, patient = patient, params = params)

		"""
			CROP IMAGES
		"""
		if params['use_cropped_images']:

			logging.debug('Start cropping images')
			# crop image
			data = create_cropped_images(data = data, meta_data = meta_data, params = params)

		"""
			SAVE DATA
		"""
		# get dynamic group name based on params
		save_group_name = get_mri_image_group_name()

		# save data to HDF5
		save_data_to_group_hdf5(group = save_group_name,
								data = data,
								data_name = patient, 
								hdf5_file = os.path.join(paths['hdf5_folder'], params['hdf5_file']),
								meta_data = meta_data, 
								overwrite = True)

def smooth_supervised_classifications(paths, params):
	"""
	Read in supervised classification data and smooth the regions with a kernel
	"""

	# create list of patients
	patients = create_patients(treatments = params['limit_treatments'], samples = params['limit_samples'], states = ['Tint'])

	for kernel_size in params['smoothing_kernels']:
			
		# loop over each patient
		for i, patient in enumerate(patients):

			logging.info(f'Processing patient {patient} {i}/{len(patients)}')
				
			# location of supervised classifications
			supervised = np.load(os.path.join(paths['supervised_folder'], f'{patient}.npy'))

			# new array to hold smoothed data
			supervised_smoothed = np.empty_like(supervised)

			# process each slice
			for mri_slice in range(supervised.shape[0]):

				supervised_smoothed[mri_slice] = smooth_image_most_occuring_value(img = supervised[mri_slice], kernel_size = kernel_size)

			# store supervised smoothed data
			save_folder = os.path.join(paths['supervised_smoothed_folder'], str(kernel_size))
			# create folder
			create_directory(save_folder)
			# save array
			np.save(file = os.path.join(save_folder, f'{patient}.npy'), arr = supervised_smoothed)


if __name__ == '__main__':

	tic, process, logging = set_start()

	# get all folder paths
	paths = get_paths()
	# get project parameters
	params = get_parameters()

	# perform group creation
	# perform_preprocessing(paths, params)

	smooth_supervised_classifications(paths, params)

	set_end(tic, process)