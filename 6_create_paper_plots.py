import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import colors
import re
import string
from scipy.stats import kruskal

from functions.helper_functions import set_start, read_directory, create_directory
from functions.project_functions import  get_parameters, get_paths, create_patients, check_mri_slice_validity, get_mri_image_group_name, \
										treatment_to_title, get_gradcam_methods, get_drip_loss, cnn_to_title, gradcam_to_title, get_datasets_paths, \
										get_supervised_classification, sort_model_subfolders, sig_to_character, sig_to_text
from functions.hdf5_functions import read_dataset_from_group
from functions.img_functions import mask_image
from functions.data_functions import calculate_ci_interval, mean_confidence_interval

def plot_image_per_class():

	# group name to extract images from
	group_name = 'bg_True_crop_False_gray_False'
	# group_name = 'bg_True_crop_False_gray_True_adap_histogram'

	# create list of patient names for filtering
	patients_5 = create_patients(treatments = ['1'], states = ['Tint'])[:10]
	patients_20 = create_patients(treatments = ['2'], states = ['Tint'])[:10]
	patients_40 = create_patients(treatments = ['3'], states = ['Tint'])[:10]
	patients_fresh = create_patients(states = ['fersk'])[:10]
	patients = zip(patients_5, patients_20, patients_40, patients_fresh)

	# define titles
	titles = [u'-5 °C', u'-20 °C', u'-40 °C', 'Fresh']

	for p_ind, patient_combo in enumerate(patients):

		for mri_slice in range(15,35):

			fig, axs = plt.subplots(2, 2, figsize = (8,8))
			axs = axs.ravel()

			for i, patient in enumerate(patient_combo):

				# read fresh image
				imgs = read_dataset_from_group(group_name = group_name, dataset = patient, hdf5_file = os.path.join(paths['hdf5_folder'], params['hdf5_file']))
				# take out slice
				img = imgs[mri_slice]
				#plot slice
				axs[i].imshow(img, cmap = 'gray')
				# set title
				axs[i].set_title(titles[i])
				# remove axis
				axs[i].axis('off')


			# create plotfolder subfolder
			plot_sub_folder = os.path.join(paths['paper_plot_folder'], 'plot_image_per_class')
			# create folder if not exists
			create_directory(plot_sub_folder)

			# crop white space
			fig.set_tight_layout(True)
			# save the figure
			fig.savefig(os.path.join(plot_sub_folder, f'plot_image_per_class_{p_ind}_{mri_slice}.jpg'))

			# close the figure environment
			plt.close()
			


def model_performance_table():

	# model folder
	model_folder = os.path.join(paths['model_folder'], get_mri_image_group_name())
	# read model subfolders to process
	model_subfolders = [x for x in read_directory(directory = model_folder, subfolders = False) if x != '.DS_Store']
	# sort model subfolder
	model_subfolders = sort_model_subfolders(model_subfolders)
	# table with training results
	df = pd.DataFrame()

	# folder where to save datasets to
	dataset_folder = os.path.join(paths['dataset_folder'], get_mri_image_group_name())

	# read datasets from file
	datasets = get_datasets_paths(dataset_folder)

	# training samples
	n_train = np.load(datasets['Y_train_aug']).shape[0] if params['use_grayscale_augmented_images_for_training'] else np.load(datasets['Y_train']).shape[0]
	n_val = np.load(datasets['Y_val']).shape[0]
	n_test = np.load(datasets['Y_test']).shape[0]
	
	for model_subfolder in model_subfolders:
		
		# read model name from subfolder
		cnn_model = cnn_to_title(re.search('.*?_(.*)', model_subfolder).group(1))
				
		# get training history file 
		training_history = os.path.join(paths['model_folder'], get_mri_image_group_name(), model_subfolder, 'history_training.csv')
		# get test history file
		test_history = os.path.join(paths['model_folder'], get_mri_image_group_name(), model_subfolder, 'history_test.csv')
		# load training data as dataframe
		training_data = pd.read_csv(training_history, index_col = 0)
		# load test data as dataframe
		test_data = pd.read_csv(test_history, index_col = 0)

		test_accuracy = test_data.loc['test_accuracy'].iloc[0]
		test_loss = test_data.loc['test_loss'].iloc[0]

		# get data from final epoch from early stopping
		final_epoch = training_data.sort_values('val_loss').iloc[0]

		# accuracy confidence interval
		training_accuracy_ci = calculate_ci_interval(value = final_epoch['accuracy'], n = n_train)
		val_accuracy_ci = calculate_ci_interval(value = final_epoch['val_accuracy'], n = n_val)
		test_accuracy_ci = calculate_ci_interval(value = test_accuracy, n = n_test)

		df_add = {	#'Training Loss' : final_epoch['loss'],
					'Training Accuracy' : final_epoch['accuracy'],
					'Training CI' :  training_accuracy_ci,
					#'Validation Loss' : final_epoch['val_loss'],
					'Validation Accuracy' : final_epoch['val_accuracy'],
					'Validation CI' :  val_accuracy_ci,
					#'Test Loss' : test_loss,
					'Test Accuracy' : test_accuracy,
					'Test CI' :  test_accuracy_ci,
					'Epoch' : int(final_epoch.name)}

		df[cnn_model] = pd.Series(df_add)

	# table with rounded numbers	
	df = df.T.round(2)

	# latex table
	df_latex = pd.DataFrame()
	
	for x in ['Training', 'Validation', 'Test']:
		# latex table
		df_latex[f'{x} accuracy'] = df[f'{x} Accuracy'].astype(str) + u' (\u00B1' + df[f'{x} CI'].astype(str) + ')'
	df_latex.insert(0, 'Epoch', df['Epoch'].astype(int))

	# create plotfolder subfolder
	save_folder = os.path.join(paths['paper_plot_folder'], 'model_performance')
	# create folder if not exists
	create_directory(save_folder)
	
	# save latex table to file
	df_latex.to_excel(os.path.join(save_folder, 'model_performance.xls'))


def plot_training_results_per_epoch():

	# append dataset name to model folder
	model_folder = os.path.join(paths['model_folder'], get_mri_image_group_name())
	# read model subfolders to process
	model_subfolders = [x for x in read_directory(directory = model_folder, subfolders = False) if x != '.DS_Store']
	# sort model subfolders
	model_subfolders = sort_model_subfolders(model_subfolders)

	# set up plot environment
	fig, axs = plt.subplots(len(model_subfolders), 2, figsize = (20,10))
	# axs = axs.ravel()

	# plot each model
	for i, model_subfolder in enumerate(model_subfolders):

		# get training history file 
		training_history = os.path.join(paths['model_folder'], get_mri_image_group_name(), model_subfolder, 'history_training.csv')
		# get test history file
		test_history = os.path.join(paths['model_folder'], get_mri_image_group_name(), model_subfolder, 'history_test.csv')

		# load training data as dataframe
		data = pd.read_csv(training_history, index_col = 0)
		
		# load test data as dataframe
		test_data = pd.read_csv(test_history, index_col = 0)

		test_accuracy = round(test_data.loc['test_accuracy'].iloc[0],3)
		test_loss = round(test_data.loc['test_loss'].iloc[0],3)

		# read model name from subfolder
		cnn_model = cnn_to_title(re.search('.*?_(.*)', model_subfolder).group(1))
		
		# get data from final epoch from early stopping
		final_epoch = data.sort_values('val_loss').iloc[0]

		x = range(1,len(data) + 1)

		for j, col in enumerate(['loss', 'accuracy']):

			# take every nth sample
			n = 5
			# plot loss
			axs[i,j].plot(x[::n], data[f'{col}'][::n], label = 'training (80%)', zorder = 10)
			if j == 0:
				title = '{} | Loss (crossentropy) | train loss {} | val epoch {} | test loss {}'.format(cnn_model,
																								round(final_epoch['loss'], 3) ,
																								round(final_epoch['val_loss'], 3), 
																								test_loss)
				axs[i,j].set_title(title)
			else:
				title = '{} | Accuracy | train acc {} | val acc {} | test acc {}'.format(cnn_model,
																								round(final_epoch['accuracy'], 3) ,
																								round(final_epoch['val_accuracy'], 3), 
																								test_accuracy)
				axs[i,j].set_title(title)
			axs[i,j].set_xlim(0,int(final_epoch.name))


			axs[i,j].set_xlabel('number of epochs')
			axs[i,j].plot(x[::n], data[f'val_{col}'][::n], label = 'validation (10%)')
			# create vertical line
			# axs[i,j].axvline(int(final_epoch.name) + 1, linestyle = '--', color = 'gray')

			
		# customize plot
		for ax in axs.ravel():

			ax.legend(loc='best', prop={'size': 10})
			# ax.set_xlim(0, final_epoch)
			


	# create plotfolder subfolder
	plot_sub_folder = os.path.join(paths['paper_plot_folder'], 'training_results', get_mri_image_group_name())
	# create folder if not exists
	create_directory(plot_sub_folder)

	# crop white space
	fig.set_tight_layout(True)
	# save the figure
	fig.savefig(os.path.join(plot_sub_folder, 'training_results.pdf'))
	# close the figure environemtn
	plt.close()



def plot_iou_by_threshold():

	# list of patients
	patients = create_patients(treatments = params['limit_treatments'], samples = params['limit_samples'], states = ['Tint'])	
	# model subfolders
	model_subfolders = [x for x in read_directory(directory = os.path.join(paths['model_folder'], get_mri_image_group_name()), subfolders = False) if x != '.DS_Store']
	# sort model subfolders
	model_subfolders = sort_model_subfolders(model_subfolders)
	# read available models
	processed_models = [x for x in read_directory(directory = os.path.join(paths['table_folder'], 'activations_by_threshold', get_mri_image_group_name()), subfolders = False)]
				
	for smoothing_size in [4]:

		logging.info(f'--- Processing smoothing size: {smoothing_size} ---')

		# create plot environment
		fig, axs = plt.subplots(len(model_subfolders),3, figsize = (14,12), sharex=True, sharey=True)
		axs = axs.ravel()
		
		# set up the plot counter
		plt_cnt = 0
		for model_subfolder in model_subfolders:

			logging.info(f'=== Processing model subfolder {model_subfolder} ===')
		
			# check if model is available
			if model_subfolder not in processed_models:
				logging.warning(f'No activation threshold data for model : {model_subfolder}')
				continue

			# name of the model (minus the datetime)
			cnn_model = re.search('.*?_(.*)', model_subfolder).group(1)

			# empty dataframe to hold all data
			df = pd.DataFrame()

			# read available gradcam methods in folder
			# gradcam_methods = read_directory(os.path.join(paths['class_activation_folder'], get_mri_image_group_name(), model_subfolder), subfolders=False)
			gradcam_methods = get_gradcam_methods()

			# gradcam_methods = read_directory(os.path.join(paths['table_folder'], 'activations_by_threshold', get_mri_image_group_name(), model_subfolder), subfolders = False)
			# process all gradcam methods
			for gradcam_method in gradcam_methods:

				# location of data
				data_folder = os.path.join(paths['table_folder'], 'activations_by_threshold', get_mri_image_group_name(), model_subfolder, gradcam_method, str(smoothing_size))

				# read patients for which we have data
				available_patients = [x[:-4] for x in read_directory(data_folder, subfolders = False)]
				
				# read in data
				for patient in patients:
					# check if there is patient data
					if patient in available_patients:
						# read in data
						df_add = pd.read_csv(os.path.join(data_folder, f'{patient}.csv'))
						# add to dataframe
						df = pd.concat([df, df_add], ignore_index=True)
					else:
						logging.warning(f'No patient data for: {patient}')
				
			# new dataframe
			new_df = pd.DataFrame()		
			# get average iou per threshold, method and treatment
			for (treatment, gradcam_method, threshold), data in df.groupby(['treatment', 'gradcam_method', 'threshold']):

				# calculate average iou
				avg_iou = data['iou'].mean()
				# add to dataframe
				new_df[f'{treatment}_{gradcam_method}_{threshold}'] = pd.Series({'treatment' : treatment, 'gradcam_method' : gradcam_method, 'threshold' : threshold, 'avg_iou' : avg_iou})
			
			# transpose dataframe
			new_df = new_df.T

			for (treatment, gradcam_method), data in new_df.groupby(['treatment', 'gradcam_method']):

				ax_index = plt_cnt + (plt_cnt * 2) + (treatment - 1)
				axs[ax_index].plot(data['threshold'], data['avg_iou'], label = gradcam_to_title(gradcam_method))
				axs[ax_index].set_title(treatment_to_title(treatment))

				# extra adjustments
				if ax_index % 3 == 0:
					axs[ax_index].set_ylabel('{}IoU - {}'.format('$\mathit{m}$', cnn_to_title(cnn_model)))
		
			plt_cnt +=1

		for i, ax in enumerate(axs):
			ax.set_ylim(0,1)
			# ax.yaxis.grid()
			
			if i in range(2,100,3):
				ax.legend()
			if i >= len(axs) -3:
				ax.set_xlabel('$\mathregular{\mathit{t}_{CAM}}$')
	
		# create plotfolder subfolder
		plot_sub_folder = os.path.join(paths['paper_plot_folder'], 'plot_iou_by_threshold', get_mri_image_group_name())
		# create folder if not exists
		create_directory(plot_sub_folder)

		# crop white space
		fig.set_tight_layout(True)
		# save the figure
		fig.savefig(os.path.join(plot_sub_folder, f'plot_iou_by_threshold_{str(smoothing_size)}.pdf'))

		# close the figure environment
		plt.close()


def plot_activations_per_slice_all(mask_background = True, skip_processed_subjects = True):


	# get a list of all patients to process
	patients = create_patients(treatments = params['limit_treatments'], samples = params['limit_samples'], states = ['Tint'])
	# get model subfolders
	model_subfolders = [x for x in read_directory(directory = os.path.join(paths['model_folder'], get_mri_image_group_name()), subfolders = False) if x != '.DS_Store']
	# sort model subfolders
	model_subfolders = sort_model_subfolders(model_subfolders)

	# define gradcam methods
	gradcam_methods = get_gradcam_methods()

	# dynamically create hdf5 file
	hdf5_file = os.path.join(paths['hdf5_folder'], params['hdf5_file'])

	# processed files
	processed = read_directory(os.path.join(paths['paper_plot_folder'], 'class_activations_per_slice'))
	processed = ['{}_{}'.format(x.split(os.sep)[-2], x[-6:-4]) for x in processed]
	
	for patient in patients:

		logging.info(f'Processing patient : {patient}')

		# get original data
		original_data = read_dataset_from_group(group_name = get_mri_image_group_name(), dataset = patient, hdf5_file = hdf5_file)
		# get mask background
		mask_data = read_dataset_from_group(group_name = 'bg_True_crop_False_gray_False', dataset = patient, hdf5_file = hdf5_file)
		# load supervised classification
		supervised_ma = get_supervised_classification(smoothing_size = 12, patient = patient)

		# define the colors we want
		plot_colors = ['black', '#c08080', '#408080']
		# create a custom listed colormap (so we can overwrite the colors of predefined cmaps)
		cmap = colors.ListedColormap(plot_colors)
					
		# process slice
		for mri_slice in range(0,54):

			if check_mri_slice_validity(patient = patient, mri_slice = mri_slice, total_num_slices = 54):

				logging.info(f'Slice {mri_slice}')

				# check if already plotted
				if f'{patient}_{mri_slice}' in processed:
					logging.debug(f'Patient {patient} Slice {mri_slice} already plotted, skipping...')
					continue

				# setting up the plot environement
				fig, axs = plt.subplots(nrows = len(gradcam_methods), ncols= len(model_subfolders) + 1, figsize = (10,7))
				# title font size
				font_size = 11
				
				for i, gradcam_method in enumerate(gradcam_methods):
					
					for j, model_subfolder in enumerate(model_subfolders):

					
						# create file location
						f_cam = os.path.join(paths['class_activation_folder'], get_mri_image_group_name(), model_subfolder, gradcam_method, f'{patient}.npy')
						# load cam data
						cam = np.load(f_cam)
					
						# plot original image
						axs[i,j].imshow(original_data[mri_slice], cmap = 'gray')

						if mask_background:
							# mask cam image
							cam_no_bg = mask_image(img = cam[mri_slice], segmented_img = mask_data[mri_slice], mask_value = 0, fill_value = None)						
							#  plot class activations 
							axs[i,j].imshow(cam_no_bg, cmap = 'jet', alpha = 0.5, vmin = 0., vmax = 1.)
						else:
							axs[i,j].imshow(cam[mri_slice], cmap = 'jet', alpha = 0.5, vmin = 0., vmax = 1.)

						# read model name from subfolder
						cnn_model = re.search('.*?_(.*)', model_subfolder).group(1)
						# set to title
						cnn_title = cnn_to_title(cnn_model)

						if j == len(model_subfolders) -1 and i == 0:
							axs[i,j + 1].imshow(supervised_ma[mri_slice], cmap = cmap, interpolation = 'none')
							axs[i,j + 1].set_title('Supervised', size = font_size)

						# axs[i,j].axis('off')

						# x_axis = axs[i,j].axes.get_yaxis()
						# x_axis.set_label_text('foo')
						# x_label = x_axis.get_label()
						# x_label.set_visible(True)

						# set titles
						if i == 0:
							axs[i,j].set_title(cnn_title, size = font_size)
						if j == 0:
							axs[i,j].set_ylabel(gradcam_to_title(gradcam_method), size = font_size)#, rotation='vertical', x=-0.2,y=0.5)
	

				# make adjustments to each subplot	
				for ax in axs.ravel():
					ax.set_yticklabels([])
					ax.set_xticklabels([])
					ax.set_yticks([])
					ax.set_xticks([])
					for pos in ['right', 'left', 'top', 'bottom']:
						spines = ax.spines[pos]
						spines.set_visible(False)

				# create plotfolder subfolder
				plot_sub_folder = os.path.join(paths['paper_plot_folder'], 'class_activations_per_slice', patient)
				# create folder if not exists
				create_directory(plot_sub_folder)

				# crop white space
				fig.set_tight_layout(True)
				# save the figure
				fig.savefig(os.path.join(plot_sub_folder, f'class_activations_per_slice_{patient.replace(" ", "_")}_{mri_slice}.jpg'))

				# close the figure environment
				plt.close()


def plot_iou_with_supervised_for_threshold(smoothing_size = 4, threshold = 0.5):


	# data folder
	data_folder = os.path.join(paths['table_folder'], 'avg_iou_by_threshold', 'values')
	# file name
	f_name = f'{smoothing_size}_{threshold}.csv'
	# read data from file into dataframe
	all_data = pd.read_csv(os.path.join(data_folder, f_name), index_col = 0)

	
	fig, axs = plt.subplots(1,4, figsize = (15,5.1))
	axs = axs.ravel()

	titles = {	0 : u'-5 °C',
				1 : u'-20 °C',
				2 : u'-40 °C',}

	# plot the first three heatmaps
	for i in range(0,3):
		
		data = all_data[all_data.columns[i * 5:i * 5 + 5]]
		# rename column names
		data.rename(columns=lambda c: gradcam_to_title(c[2:]), inplace=True)
		# rename index names
		data.rename(index=lambda i: cnn_to_title(re.search('.*?_(.*)', i).group(1)), inplace=True)
		# sort index		
		data = data.sort_index()
		
		yticklabels = True if i == 0 else False

		ax = sns.heatmap(data, cmap = 'magma_r', square = True, annot = True, annot_kws= {'size' : 11}, cbar = False, ax= axs[i], vmin = 0, vmax = 1, yticklabels = yticklabels)

		axs[i].set_title(titles[i], y = -0.1)
		ax.xaxis.tick_top()
		ax.xaxis.set_label_position('top') 
		
		ax.set_xticklabels(labels = ax.get_xticklabels(), rotation = 45, ha = 'left')


	"""
		SUPERVISED DISTRIBUTION
	"""
	# read supervised data
	supervised_data = pd.read_csv(os.path.join(paths['table_folder'], 'supervised', 'tissue_distributions.csv'), index_col=0)
	# filter for damaged only
	supervised_data = supervised_data.query('state == "Tint"')
	supervised_data['treatment'] = supervised_data['treatment'].apply(lambda x: treatment_to_title(x))

	sns.histplot(data = supervised_data, x = 'rel_damaged', ax = axs[3], hue = 'treatment', bins = 50, kde = True, stat = 'percent', palette = ['#e41a1c','#4daf4a','#377eb8'])
	axs[3].set_xlabel('damaged (%)')

	# plot subfigure
	for n, ax in enumerate(axs):
		# ax.imshow(np.random.randn(10,10), interpolation='none')    
		ax.text(-0.1, 1.01, string.ascii_uppercase[n], transform = ax.transAxes, size=16, weight='bold')



	# create plotfolder subfolder
	plot_sub_folder = os.path.join(paths['paper_plot_folder'], 'plot_iou_by_threshold', get_mri_image_group_name())
	# create folder if not exists
	create_directory(plot_sub_folder)

	# crop white space
	fig.set_tight_layout(True)
	# save the figure
	fig.savefig(os.path.join(plot_sub_folder, 'plot_iou_with_supervised_for_threshold.pdf'))

	# close the figure environment
	plt.close()


def plot_correlation_with_drip_loss(paths):

	correlations = ['pearson', 'spearman']

	for cor in correlations:

		# read model subfolders to process
		model_subfolders = [x for x in read_directory(directory = os.path.join(paths['table_folder'], 'damaged_by_treshold', get_mri_image_group_name()), subfolders = False) if x != '.DS_Store']
		# sort name
		model_subfolders = sort_model_subfolders(model_subfolders)

		# create plot environment
		fig, axs = plt.subplots(2, len(model_subfolders) // 2, figsize = (13,10))
		axs = axs.ravel()

		# plot activations for each model subfolder
		for i, model_subfolder in enumerate(model_subfolders):

			logging.info(f'=== Processing model subfolder {model_subfolder} ===')
			
			# read model name from subfolder
			cnn_model = re.search('.*?_(.*)', model_subfolder).group(1)
			cnn_model = cnn_to_title(cnn_model)

			# data file location
			file = os.path.join(paths['table_folder'], f'correlation_with_driploss', get_mri_image_group_name(), 'data.csv')
		
			# read data
			plot_data = pd.read_csv(file, index_col=0)

			# filter plot data for model subfolder
			plot_data = plot_data.query(f"model_subfolder == '{model_subfolder}'")

			# update gradcam method name to alternative name
			plot_data['gradcam_method'] = plot_data['gradcam_method'].apply(lambda x: gradcam_to_title(x))
			# sort by gradcam method
			plot_data.sort_values('gradcam_method', inplace = True)
			
			# convert to float
			for col in ['threshold','pearson_r','pearson_p','spearman_r','spearman_p','s_statistic','s_pvalue','damaged','damaged_std']:
				plot_data[col] = plot_data[col].astype('float')

			# which correlation to plot
			y = 'pearson_r' if cor == 'pearson' else 'spearman_r'
			
			plot_data.rename(columns = {'gradcam_method' : 'CAM Method'}, inplace = True)

			# Plot line plot for each threshold
			sns.lineplot( x = 'threshold', y = y, hue = 'CAM Method', data = plot_data, ax = axs[i])

			# set the title
			axs[i].set_title(cnn_model)

		# plot adjustments
		for ax in axs:
			ax.set_ylim(-1,1)
			# set labels for axes
			ax.set_xlabel('{}'.format('$\mathregular{\mathit{t}_{CAM}}$'))
			ax.set_ylabel('r')
			

		# create plotfolder subfolder
		plot_sub_folder = os.path.join(paths['paper_plot_folder'], 'correlation', get_mri_image_group_name())
		# create folder if not exists
		create_directory(plot_sub_folder)

		# crop white space
		fig.set_tight_layout(True)
		# save the figure
		fig.savefig(os.path.join(plot_sub_folder, f'plot_correlation_with_drip_loss_{cor}.pdf'))

		# close the figure environment
		plt.close()



def plot_correlation_with_drip_loss_for_threshold(paths, threshold = 0.5):

	for cor in ['pearson', 'spearman']:

		# read model subfolders to process
		model_subfolders = [x for x in read_directory(directory = os.path.join(paths['table_folder'], 'damaged_by_treshold', get_mri_image_group_name()), subfolders = False) if x != '.DS_Store']
		# sort name
		model_subfolders = sort_model_subfolders(model_subfolders)

		# data file location
		file = os.path.join(paths['table_folder'], f'correlation_with_driploss', get_mri_image_group_name(), 'data.csv')

		# read data
		data = pd.read_csv(file, index_col=0)
		# filter for threshold
		data = data.query(f'threshold == {threshold}')

		plot_data = pd.DataFrame(index = model_subfolders, columns = get_gradcam_methods())
		# sig annotation
		annotate_data = pd.DataFrame(index = model_subfolders, columns = get_gradcam_methods())

		for model_subfolder, row in plot_data.iterrows():

			for gradcam_method, _ in row.items():

				# get correlation row
				correlation = data.query(f'model_subfolder == "{model_subfolder}" & gradcam_method == "{gradcam_method}" ')

				cor_value = float(correlation[f'{cor}_r'].values.squeeze())
				sig_value = float(correlation[f'{cor}_p'].values.squeeze())
				sig_character = sig_to_character(sig_value)
				custom_label = f'{round(cor_value,2)} {sig_character}'
				
				plot_data.at[model_subfolder, gradcam_method] = cor_value
				annotate_data.at[model_subfolder, gradcam_method] = custom_label

		# set values to float
		plot_data = plot_data.astype(float)
		# rename column names
		plot_data.rename(columns=lambda c: gradcam_to_title(c), inplace=True)
		# rename index names
		plot_data.rename(index=lambda i: cnn_to_title(re.search('.*?_(.*)', i).group(1)), inplace=True)
		# sort index		
		plot_data = plot_data.sort_index()

		# create figure
		fig, ax = plt.subplots(1,1, figsize = (6,6))
		# plot heatmap
		sns.heatmap(plot_data, cmap = 'bwr_r', square = True, annot = annotate_data, annot_kws= {'size' : 11, 'va' : 'center', 'ha' : 'center'}, fmt = '', cbar = False, ax = ax, vmin = -0.7, vmax = 0.7)
		
		# adjust labels
		ax.xaxis.tick_top()
		ax.xaxis.set_label_position('top') 
		ax.set_xticklabels(labels = ax.get_xticklabels(), rotation = 45, ha = 'left')
		
		# create plotfolder subfolder
		plot_sub_folder = os.path.join(paths['paper_plot_folder'], 'correlation', get_mri_image_group_name())
		# create folder if not exists
		create_directory(plot_sub_folder)

		# crop white space
		fig.set_tight_layout(True)
		# save the figure
		fig.savefig(os.path.join(plot_sub_folder, f'plot_correlation_with_drip_loss_for_threshold_{cor}_{threshold}.pdf'))

		# close the figure environment
		plt.close()
		

def plot_iou_by_cam(paths, params):

	# load in data
	data = pd.read_csv(os.path.join(paths['table_folder'], f'iou_by_cam', get_mri_image_group_name(), 'data.csv'), index_col = 0)

	# change the name of the model to something we want to have on the plot
	data['model_subfolder'] = data['model_subfolder'].apply(lambda x : cnn_to_title(re.search('.*?_(.*)', x).group(1)))
	# sort by model
	data = data.sort_values('model_subfolder')
	
	# set up the plot 
	fig, axs = plt.subplots(3,3, figsize = (10,10))
	axs = axs.ravel()

	# define thresholds
	thresholds = params['cam_thresholds']

	# plot for each threshold
	for i, threshold in enumerate(thresholds):

		# filter data
		data_filtered = data.query(f'threshold == {threshold}')
		# boxplot
		sns.boxplot(x = data_filtered['model_subfolder'], y = data_filtered['iou'], ax = axs[i])
		# set title
		axs[i].set_title('{}={}'.format('$\mathregular{\mathit{t}_{CAM}}$', threshold ))

	# adjust the plot	
	for i, ax in enumerate(axs):

		# set y axis limit
		ax.set_ylim(0,1.1)
		# remove x labels
		ax.set_xlabel('')

		if i in range(0,30,3):
			# create y labels
			ax.set_ylabel('{}IoU'.format('$\mathit{m}$'))
		else:
			# create y labels
			ax.set_ylabel(None)
		

		
		# only for the first column we want to have labels
		if i >= len(axs) - 3:
			# get labels
			labels = [x.get_text() for x in ax.get_xticklabels()]
			# extract cnn model and use conversion function\
			# labels = [cnn_to_title(re.search('.*?_(.*)', x).group(1)) for x in labels]
			# set the labels
			ax.set_xticklabels(labels, rotation = 45, ha = 'right')

			
		else:
			ax.set_xticklabels([])
			
	# create plotfolder subfolder
	plot_sub_folder = os.path.join(paths['paper_plot_folder'], 'iou', get_mri_image_group_name())
	# create folder if not exists
	create_directory(plot_sub_folder)

	# crop white space
	fig.set_tight_layout(True)
	# save the figure
	fig.savefig(os.path.join(plot_sub_folder, 'plot_iou_by_cam.pdf'))

	# close the figure environment
	plt.close()


def plot_iou_by_model(paths, params):

	# load in data
	data = pd.read_csv(os.path.join(paths['table_folder'], f'iou_by_model', get_mri_image_group_name(), 'data.csv'), index_col=0)

	# set up the plot environment
	fig, axs = plt.subplots(3,3, figsize = (10,10))
	axs = axs.ravel()

	for i, threshold in enumerate(params['cam_thresholds']):

		# filter data
		data_filtered = data.query(f'threshold == {threshold}')
		# boxplot
		sns.boxplot(y = data_filtered['gradcam_method'], x = data_filtered['iou'], ax = axs[i], orient = 'h', palette= 'Accent')
		axs[i].set_title('{}={}'.format('$\mathregular{\mathit{t}_{CAM}}$', threshold ))
	
	for i, ax in enumerate(axs):

		ax.set_xlim(0,1.1)
		ax.set_ylabel('')

		if i >= len(axs) -3:
			ax.set_xlabel('{}IoU'.format('$\mathit{m}$'))
		else:
			ax.set_xlabel(None)

		if i in range(0,30,3):
			# get labels
			labels = [x.get_text() for x in ax.get_yticklabels()]
			# extract cnn model and use conversion function\
			labels = [gradcam_to_title(x) for x in labels]
			# set the labels
			ax.set_yticklabels(labels)
		else:
			ax.set_yticklabels([])
		
	# create plotfolder subfolder
	plot_sub_folder = os.path.join(paths['paper_plot_folder'], 'iou', get_mri_image_group_name())
	# create folder if not exists
	create_directory(plot_sub_folder)

	# crop white space
	fig.set_tight_layout(True)
	# save the figure
	fig.savefig(os.path.join(plot_sub_folder, 'plot_iou_by_model.pdf'))

	# close the figure environment
	plt.close()

def plot_iou_by_model_and_cam_for_threshold(paths, params, threshold = 0.5):


	# set up the plot 
	fig, axs = plt.subplots(1,2, figsize = (10,5))
	axs = axs.ravel()

	"""
		PLOT IOU BY CAM
	"""
	# load in data
	data = pd.read_csv(os.path.join(paths['table_folder'], f'iou_by_cam', get_mri_image_group_name(), 'data.csv'), index_col = 0)
	# change the name of the model to something we want to have on the plot
	data['model_subfolder'] = data['model_subfolder'].apply(lambda x : cnn_to_title(re.search('.*?_(.*)', x).group(1)))
	# sort by model
	data = data.sort_values('model_subfolder')

	# filter data
	data_filtered = data.query(f'threshold == {threshold}')


	# test for differences in groups
	groups = [x[1]['iou'].values for x in data_filtered.groupby('model_subfolder')]
	groups = np.vstack(groups).transpose()	
	# perform kruskal
	kruskal_s, kruskal_p = kruskal(groups[:,0], groups[:,1], groups[:,2], groups[:,3], groups[:,4], groups[:,5] )

	# boxplot
	sns.boxplot(x = data_filtered['model_subfolder'], y = data_filtered['iou'], ax = axs[0])
	# set title
	axs[0].set_title('IoU across CAMs | {}={},{} | {}={}'.format('${\chi}^2$', round(kruskal_s,1), sig_to_text(kruskal_p),  '$\mathregular{\mathit{t}_{CAM}}$', threshold ))

	# calculate medians to show as annotations
	medians = data_filtered.groupby(['model_subfolder'])['iou'].median()

	# annotate above each xtick
	for xtick in axs[0].get_xticks():
		axs[0].text(xtick, medians[xtick] * 0.965, round(medians[xtick],2), horizontalalignment = 'center', size = 'x-small', color = 'white', weight = 'semibold')

	"""
		PLOT IOU BY MODEL
	"""
	# load in data
	data = pd.read_csv(os.path.join(paths['table_folder'], f'iou_by_model', get_mri_image_group_name(), 'data.csv'), index_col=0)

	# filter data
	data_filtered = data.query(f'threshold == {threshold}')

	# test for differences in groups
	groups = [x[1]['iou'].values for x in data_filtered.groupby('gradcam_method')]
	groups = np.vstack(groups).transpose()	

	# perform kruskal
	kruskal_s, kruskal_p = kruskal(groups[:,0], groups[:,1], groups[:,2], groups[:,3], groups[:,4])


	# boxplot
	sns.boxplot(x = data_filtered['gradcam_method'], y = data_filtered['iou'], ax = axs[1], palette= 'Accent')
	# axs[1].set_title('IoU across CNNs - {}={}'.format('$\mathregular{\mathit{t}_{CAM}}$', threshold ))
	# set title
	axs[1].set_title('IoU across CNNs | {}={},{} | {}={}'.format('${\chi}^2$', round(kruskal_s,1), sig_to_text(kruskal_p),  '$\mathregular{\mathit{t}_{CAM}}$', threshold ))

	# calculate medians to show as annotations
	medians = data_filtered.groupby(['gradcam_method'])['iou'].median()
	# sort medians
	medians = medians.reindex(get_gradcam_methods())
	
	# annotate above each xtick
	for xtick in axs[1].get_xticks():
		axs[1].text(xtick, medians[xtick] * 0.940, round(medians[xtick],2), horizontalalignment = 'center', size = 'x-small', color = 'black', weight = 'semibold')

	for i, ax in enumerate(axs):

		ax.set_ylim(0,1.1)
		ax.set_xlabel('')

		# get labels
		labels = [x.get_text() for x in ax.get_xticklabels()]
		# change labels for CAM methods
		if i == 1:
			labels = [gradcam_to_title(x) for x in labels]
		ax.set_xticklabels(labels, rotation = 45, ha = 'right')

		ax.set_ylabel('{}IoU'.format('$\mathit{m}$'))
		

	# plot subfigure
	for n, ax in enumerate(axs):
		# ax.imshow(np.random.randn(10,10), interpolation='none')    
		ax.text(-0.1, 1.01, string.ascii_uppercase[n], transform = ax.transAxes, size=16, weight='bold')


	# create plotfolder subfolder
	plot_sub_folder = os.path.join(paths['paper_plot_folder'], 'iou', get_mri_image_group_name())
	# create folder if not exists
	create_directory(plot_sub_folder)

	# crop white space
	fig.set_tight_layout(True)
	# save the figure
	fig.savefig(os.path.join(plot_sub_folder, f'plot_iou_by_model_and_cam.pdf'))

	# close the figure environment
	plt.close()


def plot_correlation_with_iou_for_threshold(paths, params, threshold):

	# data location
	f = os.path.join(paths['table_folder'], f'drip_loss_correlation_with_iou', get_mri_image_group_name(), 'data.csv')
	# load data
	df_cor = pd.read_csv(f, index_col=0)

	# read available models
	model_subfolders = [x for x in read_directory(directory = os.path.join(paths['table_folder'], 'damaged_by_treshold', get_mri_image_group_name()), subfolders = False) if x != '.DS_Store']
	# sort model subfolder
	model_subfolder = sort_model_subfolders(model_subfolders)
	# get gradcam methods
	gradcam_methods = get_gradcam_methods()

	# create plot here
	fig, ax = plt.subplots(1,1, figsize = (6,6))

	# read IOU data
	df_iou = pd.read_csv(os.path.join(paths['table_folder'], 'avg_iou_by_threshold', 'values', f'4_{threshold}.csv'), index_col = 0)
	# new dataframe to hold average iou value
	new_df_iou = pd.DataFrame()
	# model, gradcam -> iou
	data = pd.DataFrame()

	# get model subfolders
	for model_subfolder in model_subfolders:
		for gradcam_method in gradcam_methods:

			iou = []

			for t in [1,2,3]:

				iou.append(df_iou.loc[model_subfolder][f'{t}-{gradcam_method}'])

			new_df_iou[f'{model_subfolder}_{gradcam_method}_{threshold}'] = pd.Series({'iou' : np.mean(iou)})
			

			df_cor_filtered = df_cor.query(f"threshold == {threshold} & gradcam_method == '{gradcam_method}' & model_subfolder == '{model_subfolder}'   ")
			data = pd.concat([data, df_cor_filtered])

	
	data['iou'] = new_df_iou.T

	# rename column names
	data['gradcam_method'] = data['gradcam_method'].apply(lambda x: gradcam_to_title(x))
	data['model_subfolder'] = data['model_subfolder'].apply(lambda x: cnn_to_title(re.search('.*?_(.*)', x).group(1)))

	data.rename( columns = {'model_subfolder':'CNN Backbone', 
							'gradcam_method' : 'CAM method', 
							'spearman_r' : 'Spearman R',
							'iou' : 'mIoU'}, inplace = True)

	# create scatterplot
	sns.scatterplot(data = data, 
				x = "mIoU", 
				y = "Spearman R", 
				hue = "CNN Backbone",
				style = "CAM method", 
				# size = 10, 
				# style="gradcam_method",
				# palette=["b", "r"],
				s = 300,
				ax = ax,
				legend = 'brief', # 'brief'
				)

	h,l = ax.get_legend_handles_labels()
	# plot.axes[0].legend_.remove()
	plt.legend(h,l, ncol=2, loc = 'lower right')

	ax.set_ylim(-1,1)

	# create plotfolder subfolder
	plot_sub_folder = os.path.join(paths['paper_plot_folder'], 'correlation', get_mri_image_group_name())
	# create folder if not exists
	create_directory(plot_sub_folder)

	# crop white space
	fig.set_tight_layout(True)
	# save the figure
	fig.savefig(os.path.join(plot_sub_folder, 'correlation_iou_plot.pdf'))

	# close the figure environment
	plt.close()

def create_combined_plot(paths, params, threshold, cor = 'spearman'):

	# create plot
	fig, axs = plt.subplots(1,3, figsize = (16,6))
	axs = axs.ravel()

	# read model subfolders to process
	model_subfolders = [x for x in read_directory(directory = os.path.join(paths['table_folder'], 'damaged_by_treshold', get_mri_image_group_name()), subfolders = False) if x != '.DS_Store']
	# sort name
	model_subfolders = sort_model_subfolders(model_subfolders)


	"""
		Correlation plot first
	"""

	# data file location
	file = os.path.join(paths['table_folder'], f'correlation_with_driploss', get_mri_image_group_name(), 'data.csv')

	# read data
	data = pd.read_csv(file, index_col=0)
	# filter for threshold
	data = data.query(f'threshold == {threshold}')

	plot_data = pd.DataFrame(index = model_subfolders, columns = get_gradcam_methods())
	# sig annotation
	annotate_data = pd.DataFrame(index = model_subfolders, columns = get_gradcam_methods())

	for model_subfolder, row in plot_data.iterrows():

		for gradcam_method, _ in row.items():

			# get correlation row
			correlation = data.query(f'model_subfolder == "{model_subfolder}" & gradcam_method == "{gradcam_method}" ')

			cor_value = float(correlation[f'{cor}_r'].values.squeeze())
			sig_value = float(correlation[f'{cor}_p'].values.squeeze())
			sig_character = sig_to_character(sig_value)
			custom_label = f'{round(cor_value,2)} {sig_character}'
			
			plot_data.at[model_subfolder, gradcam_method] = cor_value
			annotate_data.at[model_subfolder, gradcam_method] = custom_label

	# set values to float
	plot_data = plot_data.astype(float)
	# rename column names
	plot_data.rename(columns=lambda c: gradcam_to_title(c), inplace=True)
	# rename index names
	plot_data.rename(index=lambda i: cnn_to_title(re.search('.*?_(.*)', i).group(1)), inplace=True)
	# sort index		
	plot_data = plot_data.sort_index()

	# plot heatmap
	sns.heatmap(plot_data, cmap = 'bwr_r', square = True, annot = annotate_data, annot_kws= {'size' : 11, 'va' : 'center', 'ha' : 'center'}, fmt = '', cbar = False, ax = axs[0], vmin = -0.7, vmax = 0.7)
	
	# adjust labels
	axs[0].xaxis.tick_top()
	axs[0].xaxis.set_label_position('top') 
	axs[0].set_xticklabels(labels = axs[0].get_xticklabels(), rotation = 45, ha = 'left')
	
	"""
		COMBINED PLOT CORRELATION WITH IOU
	"""
	# data location
	f = os.path.join(paths['table_folder'], f'drip_loss_correlation_with_iou', get_mri_image_group_name(), 'data.csv')
	# load data
	df_cor = pd.read_csv(f, index_col=0)

	# read available models
	model_subfolders = [x for x in read_directory(directory = os.path.join(paths['table_folder'], 'damaged_by_treshold', get_mri_image_group_name()), subfolders = False) if x != '.DS_Store']
	# sort model subfolder
	model_subfolder = sort_model_subfolders(model_subfolders)
	# get gradcam methods
	gradcam_methods = get_gradcam_methods()

	# read IOU data
	df_iou = pd.read_csv(os.path.join(paths['table_folder'], 'avg_iou_by_threshold', 'values', f'4_{threshold}.csv'), index_col = 0)
	# new dataframe to hold average iou value
	new_df_iou = pd.DataFrame()
	# model, gradcam -> iou
	data = pd.DataFrame()

	# get model subfolders
	for model_subfolder in model_subfolders:
		for gradcam_method in gradcam_methods:

			iou = []

			for t in [1,2,3]:

				iou.append(df_iou.loc[model_subfolder][f'{t}-{gradcam_method}'])

			new_df_iou[f'{model_subfolder}_{gradcam_method}_{threshold}'] = pd.Series({'iou' : np.mean(iou)})
			
			df_cor_filtered = df_cor.query(f"threshold == {threshold} & gradcam_method == '{gradcam_method}' & model_subfolder == '{model_subfolder}'   ")
			data = pd.concat([data, df_cor_filtered])

	data['iou'] = new_df_iou.T

	# rename column names
	data['gradcam_method'] = data['gradcam_method'].apply(lambda x: gradcam_to_title(x))
	data['model_subfolder'] = data['model_subfolder'].apply(lambda x: cnn_to_title(re.search('.*?_(.*)', x).group(1)))

	data.rename( columns = {'model_subfolder':'CNN Backbone', 
							'gradcam_method' : 'CAM method', 
							'spearman_r' if cor == 'spearman' else 'pearson_r' : 'r',
							'iou' : 'mIoU'}, inplace = True)

	data = data.sort_values('CNN Backbone')
	
	# create scatterplot
	sns.scatterplot(data = data, 
				x = "mIoU", 
				y = "r", 
				hue = "CNN Backbone",
				style = "CAM method", 
				# size = 10, 
				# style="gradcam_method",
				# palette=["b", "r"],
				s = 300,
				ax = axs[1],
				legend = True, # 'brief'
				)

	h,l = axs[1].get_legend_handles_labels()
	# plot.axes[0].legend_.remove()
	axs[1].legend(h,l, ncol=2, loc = 'upper left', fontsize = 7)

	axs[1].set_ylim(-1,1)

	"""
		DRIP LOSS PLOT
	"""
	# read data
	data = pd.read_csv(os.path.join(paths['table_folder'], 'drip_loss', 'drip_loss.csv'), delimiter = ';', decimal = ',')

	# define colors for bars
	# colors = {1 : 'red', 2 : 'green', 3 : 'blue'}
	colors = {1 : '#e41a1c', 2 : '#4daf4a', 3 : '#377eb8'}

	# group data by treatment
	for group, grouped_data in data.groupby('treatment'):

		# calculate mean score of treatment
		mean = grouped_data['thawing_loss_percentage'].mean()
		# calculate 95% confidence interval
		error = mean_confidence_interval(data = grouped_data['thawing_loss_percentage'].values)
		# create bar chart
		axs[2].bar(group, mean, width = 0.5, yerr = error[3], label = treatment_to_title(group), color = colors[group], alpha = .7)

		# turn on the legend
		axs[2].legend()
		axs[2].set_xticklabels([])
		# set y label
		axs[2].set_ylabel('Liquid Loss (%)')
	
	# plot subfigure
	for n, ax in enumerate(axs):
		# ax.imshow(np.random.randn(10,10), interpolation='none')    
		ax.text(-0.1, 1.1, string.ascii_uppercase[n], transform = ax.transAxes, size=16, weight='bold')

	# create plotfolder subfolder
	plot_sub_folder = os.path.join(paths['paper_plot_folder'], 'correlation', get_mri_image_group_name())
	# create folder if not exists
	create_directory(plot_sub_folder)

	# crop white space
	fig.set_tight_layout(True)
	# save the figure
	fig.savefig(os.path.join(plot_sub_folder, f'combined_correlation_iou_plot_{cor}.pdf'))

	# close the figure environment
	plt.close()


	
if __name__ == '__main__':

	tic, process, logging = set_start()

	# get all folder paths
	paths = get_paths()
	# get project parameters
	params = get_parameters()

	# create example images of fresh, -5C, -20C, and -40C
	# plot_image_per_class()

	# create table with model performance
	# model_performance_table()

	# plot training results per epoch for training and validation set
	# plot_training_results_per_epoch()

	# correlation with supervised classification for multiple threshold values
	# plot_iou_by_threshold()

	# plot class activiation per model and per cam method
	# plot_activations_per_slice_all()
	
	# plot iou with supervised classification for a specific threshold
	# plot_iou_with_supervised_for_threshold()

	# plot correlation class activation with drip loss
	# plot_correlation_with_drip_loss(paths = paths)

	# plot correlation class activation with drip loss for a certain threshold value
	# plot_correlation_with_drip_loss_for_threshold(paths = paths)

	# plot intersection over union by comparing different CAM methods with each other
	# plot_iou_by_cam(paths = paths, params = params)

	# plot intersection over union by comparing different CNN models
	# plot_iou_by_model(paths = paths, params = params)

	# plot intersection over union combined for models and CAM methods
	plot_iou_by_model_and_cam_for_threshold(paths = paths, params = params, threshold = 0.5)

	# plot correlation with drip loss in relation to iou with supervised classification
	# plot_correlation_with_iou_for_threshold(paths, params, threshold = 0.5)

	# combined plot with correlation with drip loss and another plot showing correlation and iou combined
	# create_combined_plot(paths, params, threshold = 0.5)

	