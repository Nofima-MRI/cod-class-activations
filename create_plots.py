import os
import numpy as np
import matplotlib.pyplot as plt

from functions.helper_functions import set_start, read_directory, create_directory
from functions.project_functions import get_parameters, get_paths, get_range_slices, create_patients, check_mri_slice_validity, get_mri_image_group_name, get_gradcam_methods
from functions.hdf5_functions import read_dataset_from_group, get_all_groups_hdf5
from functions.img_functions import mask_image

def plot_images_from_group(paths, params, skip_already_plotted = True):

	# dynamically create hdf5 file
	hdf5_file = os.path.join(paths['hdf5_folder'], params['hdf5_file'])

	# create list of patient names for filtering
	patients = create_patients(treatments = params['limit_treatments'], samples = params['limit_samples'], states = params['limit_states'])

	# read all groups in hdf5 file
	groups = get_all_groups_hdf5(hdf5_file = hdf5_file)

	# already plotted groups
	already_plotted_groups = read_directory(directory= os.path.join(paths['plot_folder'], 'groups'), subfolders = False)

	# plot all groups
	for group in groups:

		logging.info(f'Processing group {group}')

		if skip_already_plotted:
			if group in already_plotted_groups:
				logging.info(f'Group {group} already plotted, skipping...')
				continue

		# plot per patient
		for patient in patients:

			# read images
			images = read_dataset_from_group(group_name = group, dataset = patient, hdf5_file = hdf5_file)

			# setting up the plot environment
			fig, axs = plt.subplots(6, 9, figsize = (15, 10))
			axs = axs.ravel()

			# plot each image slice
			for i, mri_slice in enumerate(range(images.shape[0])):

				# check if slice is taken into account
				valid_slice = False
				if mri_slice in get_range_slices(mri_data = images):
					# check validity of mri slice
					if check_mri_slice_validity(patient = patient, mri_slice = mri_slice, total_num_slices = images.shape[0]):
						valid_slice = True

				# plot image
				axs[i].imshow(images[mri_slice], cmap = 'gray')
				axs[i].set_title(mri_slice, color = 'green' if valid_slice else 'red')
			

			# make adjustments to each subplot	
			for ax in axs:
				ax.axis('off')

			# create plotfolder subfolder
			plot_sub_folder = os.path.join(paths['plot_folder'], 'groups', group)
			# create folder if not exists
			create_directory(plot_sub_folder)

			# crop white space
			fig.set_tight_layout(True)
			# save the figure
			fig.savefig(os.path.join(plot_sub_folder, f'{patient}.png'))

			# close the figure environment
			plt.close()


def plot_activations_by_patient(paths, params, mask_background = True, skip_processed_plots = True):


	# append dataset name to model folder
	model_folder = os.path.join(paths['model_folder'], get_mri_image_group_name())
	# read model subfolders to process
	model_subfolders = [x for x in read_directory(directory = model_folder, subfolders = False) if x != '.DS_Store']
	# dynamically create hdf5 file
	hdf5_file = os.path.join(paths['hdf5_folder'], params['hdf5_file'])

	# create list of patient names for filtering
	patients = create_patients(treatments = params['limit_treatments'], samples = params['limit_samples'], states = ['Tint'])

	# plot activations for each model subfolder
	for model_subfolder in model_subfolders:

		for gradcam_method in get_gradcam_methods():

			# read class activations file location
			F = read_directory(os.path.join(paths['class_activation_folder'], get_mri_image_group_name(), model_subfolder, gradcam_method))

			# read processed plots
			processed_patients = read_directory(os.path.join(paths['plot_folder'], 'class_activations', get_mri_image_group_name(), model_subfolder, gradcam_method), subfolders = False, return_none_if_not_exists = True)
			if processed_patients is None:
				processed_patients = []
			else:
				processed_patients = [x[:-4] for x in processed_patients]

			# get data for each patient
			for f in F:
				# extract patient name from file
				patient = f.split(os.sep)[-1][:-4]
				
				# check if patient has already been processed
				if skip_processed_plots:
					if patient in processed_patients:
						logging.debug(f'patient {patient} already plotted, skipping')
						continue

				# read in activations
				cam = np.load(f)

				# check if patient needs to be plotted
				if patient in patients:

					logging.info(f'Processing patient : {patient}')

					# get original data
					data = read_dataset_from_group(group_name = get_mri_image_group_name(), dataset = patient, hdf5_file = hdf5_file)
					# get mask background
					mask_data = read_dataset_from_group(group_name = 'bg_True_crop_False_gray_False', dataset = patient, hdf5_file = hdf5_file)

					# setting up the plot environment
					fig, axs = plt.subplots(5, 8, figsize = (15, 10))
					axs = axs.ravel()

					# process each slice
					for i, mri_slice in enumerate(get_range_slices(data)):

						# check validity of mri slice
						if check_mri_slice_validity(patient = patient, mri_slice = mri_slice, total_num_slices = data.shape[0]):

							# plot original image
							axs[i].imshow(data[mri_slice], cmap = 'gray')

							if mask_background:
								# mask cam image
								cam_no_bg = mask_image(img = cam[mri_slice], segmented_img = mask_data[mri_slice], mask_value = 0, fill_value = None)						
								#  plot class activations 
								axs[i].imshow(cam_no_bg, cmap = 'jet', alpha = 0.5, vmin = 0., vmax = 1.)
							else:
								axs[i].imshow(cam[mri_slice], cmap = 'jet', alpha = 0.5, vmin = 0., vmax = 1.)
								
							# set titles
							axs[i].set_title(mri_slice)

					# make adjustments to each subplot	
					for ax in axs:
						ax.axis('off')

					# create plotfolder subfolder
					plot_sub_folder = os.path.join(paths['plot_folder'], 'class_activations', get_mri_image_group_name(), model_subfolder, gradcam_method)
					# create folder if not exists
					create_directory(plot_sub_folder)

					# crop white space
					fig.set_tight_layout(True)
					# save the figure
					fig.savefig(os.path.join(plot_sub_folder, f'{patient}.png'))

					# close the figure environment
					plt.close()

if __name__ == "__main__":


	tic, process, logging = set_start()

	# get all folder paths
	paths = get_paths()
	# get project parameters
	params = get_parameters()

	# 1) plot images from HDF5 group
	plot_images_from_group(paths, params)

	# 2) plot class activations per sample/patient
	plot_activations_by_patient(paths, params)
