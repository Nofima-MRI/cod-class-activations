# IMPORT PACKAGES
import os
import re
import pydicom
import shutil

# IMPORT FUNCTIONS
from functions.helper_functions import set_start, set_end, read_directory, create_directory
from functions.project_functions import get_parameters, get_paths, get_protocol_translation, create_class_label, check_validity_mri_scan
from functions.hdf5_functions import save_data_to_group_hdf5, delete_group

def clear_hdf5_file(paths, params):
	"""
	Remove unwanted HDF5 groups
	"""

	# full file location of hdf5 file
	hdf5_file = os.path.join(paths['hdf5_folder'], params['hdf5_file'])

	# groups to remove
	unwanted_groups = ['CROPPED-MRI', 'NO-BG-GRAY-ADJUSTED-HISTOGRAM-MRI', 'NO-BG-GRAY-ADJUSTED-MRI', 'NO-BG-GRAY-ADJUSTED-MRI-FIXED', 'NO-BG-GRAY-ADJUSTED-MRI-PERCENTILE', 'NO-BG-MRI']

	# start removing groups
	for group in unwanted_groups:

		# delet group (no warning!!!)
		delete_group(group_name= group, hdf5_file = hdf5_file)



def process_convert_dcm_to_hdf5(paths, params, copy_dcm_files = False):
	"""
	Read dcm files from file and extract image data and save as HDF5

	Parameters
	----------
	
	"""

	# read all dicom files
	F = [f for f in read_directory(paths['mri_folder']) if f[-4:] == '.dcm']

	# loop over each file, read dicom files, save data
	for f in F:

		# verbose
		logging.info(f'Processing file : {f}')

		# read dcm file
		dataset = pydicom.dcmread(f)

		# construct meta data
		meta_data = {	'SOPClassUID' : dataset.SOPClassUID,
						'SeriesInstanceUID' : dataset.SeriesInstanceUID,
						'PatientName' : dataset.PatientName,
						'PatientID' : dataset.PatientID,
						'SeriesNumber' : dataset.SeriesNumber,
						'Rows' : dataset.Rows,
						'Columns' : dataset.Columns,
						'AcquisitionDateTime' : dataset.AcquisitionDateTime,
						'ProtocolName' : dataset.ProtocolName,
						'SeriesDescription' : dataset.SeriesDescription
					}
						
		# convert all meta data to string
		meta_data = {key : str(value) for key, value in meta_data.items()}

		# get the MRI image				
		image = dataset.pixel_array

		# verbose
		logging.debug(f'Image shape : {image.shape}')
		logging.debug(f'Image datatype : {image.dtype}')

		# if image has 3 slices, then this is the scout image (the first image to get a quick scan of the sample), this we skip
		if image.shape[0] == 3:
			logging.debug('MRI image is scout, skipping...')
			continue


		# get treatment-sample combination
		treatment_sample = re.findall('[0-9]+-[0-9]+', meta_data['PatientName'])[0]

		# get state
		state = f.split(os.sep)[-6]

		# change patient name into specific format that will be used throughout all analysis
		# This is: Torsk [treatment]-[sample] [Tint|fersk]
		# for example: Tork 1-1 fersk
		patient_name = 'Torsk {} {}'.format(treatment_sample, state)

		# check if patient scan is valid
		if not check_validity_mri_scan(patientname = patient_name, datetime = meta_data['AcquisitionDateTime']):
			logging.debug('Scan was invalid, skipping...')
			continue

		if copy_dcm_files:
			# copy .DCM file to folder with new patient name
			destination_folder = os.path.join(paths['dcm_folder'], state, patient_name)
			# create state folder
			create_directory(destination_folder)
			# define source file
			source_file = f
			# define destination file
			destination_file = os.path.join(destination_folder, f'{patient_name}.dcm')
			# start copying
			try:
				shutil.copy(source_file, destination_file)
			except Exception as e:
				logging.error(f'Failed copying .DCM file to dcm folder: {e}')
		
		# add extra meta data
		meta_data['ClassLabel'] = create_class_label(patient_name)
		meta_data['ProtocolTranslation'] = get_protocol_translation(dataset.ProtocolName)
		meta_data['DrippLossPerformed'] = False 
						
		# save data to HDF5
		save_data_to_group_hdf5(group = params['group_original_mri'],
								data = image,
								data_name = patient_name, 
								hdf5_file = os.path.join(paths['hdf5_folder'], params['hdf5_file']),
								meta_data = meta_data, 
								overwrite = True)
		
# SCRIPT STARTS HERE	
if __name__ == '__main__':

	tic, process, logging = set_start()

	# get all folder paths
	paths = get_paths()
	# get project parameters
	params = get_parameters()

	"""
		CALL FUNCTIONS
	"""

	# delete unwanted groups from HDF5 file (in case of redoing the full analyses)
	# clear_hdf5_file(paths, params)

	# read dcm files, extract meta data, extract MRI image data, and save as HDF5
	process_convert_dcm_to_hdf5(paths = paths, params = params, copy_dcm_files = True)

	set_end(tic, process)
