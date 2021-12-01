import os
import matplotlib.pyplot as plt

from functions.helper_functions import set_start
from functions.project_functions import get_paths, get_parameters, create_patients
from functions.hdf5_functions import read_dataset_from_group, read_metadata_from_group_dataset, save_meta_data_to_group_dataset

def onclick(event):
	"""
	Handles click events in interactive matplotlib environment
	"""

	# if the mouse left button is clicked
	if int(event.button) == 1:

		# get the matplotlib x data 
		xdata = event.xdata
		ydata = event.ydata

		# plot a circle on top of the plot
		square = plt.Rectangle([xdata - (params['crop_dimensions'][1] // 2 ), ydata - (params['crop_dimensions'][1] // 2)], params['crop_dimensions'][1], params['crop_dimensions'][0], color = 'red', joinstyle = 'round', alpha = 0.6)
		# add to axis
		ax.add_artist(square)
		# draw on top of plot
		fig.canvas.draw()
		# add clicks
		clicks.append([xdata, ydata])


if __name__ == "__main__":

	tic, process, logging = set_start()

	# get all folder paths
	paths = get_paths()
	# get project parameters
	params = get_parameters()

	# create list of patient names for filtering
	patients = create_patients(treatments = params['limit_treatments'], samples = params['limit_samples'], states = params['limit_states'])

	# dynamically create hdf5 file
	hdf5_file = os.path.join(paths['hdf5_folder'], params['hdf5_file'])

	# read images, and create cropped image
	for patient in patients:

		# read original MRI image
		original_mri = read_dataset_from_group(group_name = params['group_original_mri'], dataset = patient, hdf5_file = hdf5_file)
		# read in meta data
		meta_data = read_metadata_from_group_dataset(group_name = params['group_original_mri'], dataset = patient, hdf5_file = hdf5_file)

		# get middle image
		middle_slice = original_mri.shape[0] // 2

		# to store click coordinates
		clicks = []

		# setting up the plot environment
		fig, ax = plt.subplots(1,1, figsize=(10, 10))
			
		# plot image
		ax.imshow(original_mri[middle_slice], cmap = 'gray')
		ax.set_title(f'{patient} - slice {middle_slice}')

		# handler for the mouse click
		fig.canvas.mpl_connect('button_press_event', onclick)

		plt.show()
		plt.close()

		# unpack click
		if clicks:
			x, y = clicks[-1] 

			# add to meta_data
			meta_data['midpoint_x'] = x
			meta_data['midpoint_y'] = y

			logging.info(f'Clicked on x: {x}, y: {y}')

			# save clicks to hdf5
			save_meta_data_to_group_dataset(group_name = params['group_original_mri'], dataset = patient, meta_data = meta_data, hdf5_file = hdf5_file)
		


	