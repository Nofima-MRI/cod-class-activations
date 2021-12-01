import numpy as np
from sklearn.cluster import KMeans
from skimage.filters.rank import majority
from skimage.morphology import square
from skimage import exposure

def perform_knn_segmentation(n_clusters, img):
	"""
	Perform image segmentation with knn

	Parameters
	----------
	n_clusters : int
		number of clusters to segment image into
	img : np.array()
		image as numpy array
	n_jobs = int (optional)
		perform knn with multiple threads. Default set to number of CPU cores
	
	Returns
	---------
	segmented_img : np.array()
		image where each pixel is now assigned to a centroid class
	"""

	# create Kmeans model
	model = KMeans(n_clusters = n_clusters)
	
	# fit model onto flattened data
	model.fit(img.reshape(-1,1))

	# segment the image
	segmented_img = model.cluster_centers_[model.labels_]

	# convert segmented image back to original shape
	segmented_img = segmented_img.reshape(img.shape)

	return segmented_img

def mask_image(img, segmented_img, mask_value, fill_value = None):

	"""
	Mask image based on mask value. If fill_value is set, replace masked values with fill_value

	Parameters
	----------
	img : np.array()
		image as numpy array
	mask_value: int/float
		which value to mask
	fill_value : int/float (optional)
		to replace masked value with fill value. Default is None, which means no filling

	Returns
	--------
	masked_image : np.array()
		masked image
	"""

	# create a mask based on mask value
	mask = (segmented_img == mask_value)

	# create a masked image
	masked_image = np.ma.masked_array(img, mask = mask)
	
	# fill masked value with fill_value
	if fill_value is not None:
		masked_image = np.ma.filled(masked_image, fill_value = fill_value)

	return masked_image
	
def change_img_contrast(img, phi = 5, theta = 1):
	"""
		Change contrast of image
	"""

	# get the maximum intensity
	max_intensity = np.max(img)
	# adjust image
	img_adjusted = (max_intensity / phi) * (img / (max_intensity / theta))**0.5

	return img_adjusted

def sliding_window_view(arr, window_shape, steps):
	""" Produce a view from a sliding, striding window over `arr`.
		The window is only placed in 'valid' positions - no overlapping
		over the boundary.

		Credits by:
		https://gist.github.com/meowklaski/4bda7c86c6168f3557657d5fb0b5395a

		Parameters
		----------
		arr : numpy.ndarray, shape=(...,[x, (...), z])
			The array to slide the window over.

		window_shape : Sequence[int]
			The shape of the window to raster: [Wx, (...), Wz],
			determines the length of [x, (...), z]

		steps : Sequence[int]
			The step size used when applying the window
			along the [x, (...), z] directions: [Sx, (...), Sz]

		Returns
		-------
		view of `arr`, shape=([X, (...), Z], ..., [Wx, (...), Wz])
			Where X = (x - Wx) // Sx + 1

		Notes
		-----
		In general, given
		  `out` = sliding_window_view(arr,
									  window_shape=[Wx, (...), Wz],
									  steps=[Sx, (...), Sz])

		   out[ix, (...), iz] = arr[..., ix*Sx:ix*Sx+Wx,  (...), iz*Sz:iz*Sz+Wz]

		 Examples
		 --------
		 >>> import numpy as np
		 >>> x = np.arange(9).reshape(3,3)
		 >>> x
		 array([[0, 1, 2],
				[3, 4, 5],
				[6, 7, 8]])

		 >>> y = sliding_window_view(x, window_shape=(2, 2), steps=(1, 1))
		 >>> y
		 array([[[[0, 1],
				  [3, 4]],

				 [[1, 2],
				  [4, 5]]],


				[[[3, 4],
				  [6, 7]],

				 [[4, 5],
				  [7, 8]]]])
		>>> np.shares_memory(x, y)
		 True

		# Performing a neural net style 2D conv (correlation)
		# placing a 4x4 filter with stride-1
		>>> data = np.random.rand(10, 3, 16, 16)  # (N, C, H, W)
		>>> filters = np.random.rand(5, 3, 4, 4)  # (F, C, Hf, Wf)
		>>> windowed_data = sliding_window_view(data,
		...                                     window_shape=(4, 4),
		...                                     steps=(1, 1))

		>>> conv_out = np.tensordot(filters,
		...                         windowed_data,
		...                         axes=[[1,2,3], [3,4,5]])

		# (F, H', W', N) -> (N, F, H', W')
		>>> conv_out = conv_out.transpose([3,0,1,2])
		 """
	import numpy as np
	from numpy.lib.stride_tricks import as_strided
	in_shape = np.array(arr.shape[-len(steps):])  # [x, (...), z]
	window_shape = np.array(window_shape)  # [Wx, (...), Wz]
	steps = np.array(steps)  # [Sx, (...), Sz]
	nbytes = arr.strides[-1]  # size (bytes) of an element in `arr`

	# number of per-byte steps to take to fill window
	window_strides = tuple(np.cumprod(arr.shape[:0:-1])[::-1]) + (1,)
	# number of per-byte steps to take to place window
	step_strides = tuple(window_strides[-len(steps):] * steps)
	# number of bytes to step to populate sliding window view
	strides = tuple(int(i) * nbytes for i in step_strides + window_strides)

	outshape = tuple((in_shape - window_shape) // steps + 1)
	# outshape: ([X, (...), Z], ..., [Wx, (...), Wz])
	outshape = outshape + arr.shape[:-len(steps)] + tuple(window_shape)
	return as_strided(arr, shape=outshape, strides=strides, writeable=False)

def calculate_midpoint(img, threshold = 1):
	"""
	Calculate the center (midpoint) of the object within an image with background
	"""

	# check which rows have an object
	rows = np.argwhere(np.any(img[:] >= threshold, axis = 1))
	# check which colums have an annotation
	cols = np.argwhere(np.any(img[:] >= threshold, axis = 0))
	# get start and stop rows 
	min_rows, max_rows = rows[0][0], rows[-1][0]
	# get start and stop columns
	min_cols, max_cols = cols[0][0], cols[-1][0]

	# calculate the center of the min and max
	mid_rows = (min_rows + max_rows) // 2
	mid_cols = (min_cols +  max_cols) // 2

	# return midpoint
	return mid_cols, mid_rows

def perform_piecewise_linear_transformation(pix, r1, s1, r2, s2, max_value = 2**12-1):
	"""
	Function to map each intensity level to output intensity level.

	Contrast stretching

	https://www.geeksforgeeks.org/python-intensity-transformation-operations-on-images/
	"""
	if (0 <= pix and pix <= r1):
		return (s1 / r1)*pix
	elif (r1 < pix and pix <= r2):
		return ((s2 - s1)/(r2 - r1)) * (pix - r1) + s1
	else:
		return ((max_value - s2)/(max_value - r2)) * (pix - r2) + s2

def perform_histogram_equalization(img, max_value = 2**12, i = 0):
	"""
	
	https://docs.opencv.org/master/d5/daf/tutorial_py_histogram_equalization.html
	"""

	# create histogram from image
	hist,_ = np.histogram(a = img.flatten(), bins = max_value, range = [0,max_value])
	# get probability density function
	cdf = hist.cumsum()

	# create masked array of cdf and replace all 0 with a mask
	cdf_m = np.ma.masked_equal(cdf, 0)
	# normalize
	cdf_m = (cdf_m - cdf_m.min())* max_value / (cdf_m.max()-cdf_m.min())
	# fill mask back with zero again
	cdf = np.ma.filled(cdf_m,0)

	# match pixel to histogram
	return cdf[img], i

def perform_adaptive_histogram_equalization(img, kernel_size = None, clip_limit = 0.01, nbins = 256, i = 0):
	"""
	Contrast Limited Adaptive Histogram Equalization (CLAHE).

	An algorithm for local contrast enhancement, that uses histograms computed over different tile regions of the image. Local details can therefore be enhanced even in regions that are darker or lighter than most of the image.

	https://scikit-image.org/docs/dev/api/skimage.exposure.html#skimage.exposure.equalize_adapthist
	
	Parameters
	--------------
	img : ndarray
		Input image.
	kernel_size : int or array_like, optional
		Defines the shape of contextual regions used in the algorithm. If iterable is passed, it must have the same number of elements as image.ndim 
		(without color channel). If integer, it is broadcasted to each image dimension. By default, kernel_size is 1/8 of image height by 1/8 of its width.
	clip_limit : float, optional
		Clipping limit, normalized between 0 and 1 (higher values give more contrast).
	nbins : int, optional
		Number of gray bins for histogram (“data range”).
	
	Returns
	------------
	out : (N1, …,NN[, C]) ndarray
		Equalized image with float64 dtype.
	"""

	out =  exposure.equalize_adapthist(image = img, kernel_size = kernel_size, clip_limit = clip_limit, nbins = nbins)

	return out, i


def smooth_image_most_occuring_value(img, kernel_size):
	"""
	Smooth out an image by a sliding window and keeping only the most occuring value.

	Parameters
	----------
	img : np.array()
		2D numpy array
	kernel_size	: int
		size of kernel, for int = 3 then the kernel is (3,3)

	returns
	---------
	img : np.array()
		smoothed image array
	"""

	return majority(img, square(kernel_size))
