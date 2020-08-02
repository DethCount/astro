# pip install matplotlib astropy
# python planetary_crop.py

import os, time, random
import numpy as np

from PIL import Image

import matplotlib.pyplot as plt
from astropy.visualization import astropy_mpl_style
plt.style.use(astropy_mpl_style)

#configuration
input_dir = 'in' # put fits here
output_dir = 'out' # get generated images here 
interactive = False # show interface
normalize_channels = False # same max value for color channels
min_avg_color = 0.6 # target covers at least 60% of final images
min_color_stable = False # keep same min_color between images
min_color_init = 10
min_color_step = 5
padding = 10 # padding arround target
generate_differential_img = True
stack = True
stack_filename = 'stacked.' + str(time.time()) + '.png'

# internal vars
min_color_calculated = False 
color_channel_values = range(255);
previous_img = None
previous_filename = None
previous_extension = None
stack_shape = None
nb_stacked_images = 0
stack_img = None
stack_file_path = output_dir + '\\' + stack_filename
t = 'uint8'

for filename in os.listdir(input_dir):
	image_data = None
	_,extension = os.path.splitext(filename)

	if filename.endswith(".fit") or filename.endswith(".fits"):
		from astropy.utils.data import get_pkg_data_filename
		from astropy.io import fits
		image_file = get_pkg_data_filename(input_dir + '\\' + filename)
		fits.info(image_file)
		image_data = fits.getdata(image_file, ext=0)
		image_data = image_data.transpose((1,2,0))
		image_data = np.flip(image_data, axis=0)
		t = image_data.dtype
	else:
		image_data = np.asarray(Image.open(input_dir + '\\' + filename))
		print(filename)
		print(image_data.shape)

	if image_data is not None:
		if interactive:
			nb_plots = 2
			if generate_differential_img:
				nb_plots += 2

			_,splt = plt.subplots(nb_plots)

		min_color = min_color if min_color_calculated and min_color_stable else min_color_init

		# find planetary object and crop
		while True:
			image_rgb = image_data
			image_rgb = np.sum(image_rgb, axis=2)

			image_rgb[image_rgb < min_color] = 0
			image_rgb[image_rgb >= min_color] = 1
			visible = image_rgb.nonzero();

			xmin0 = visible[0].min()
			xmax0 = visible[0].max()
			ymin0 = visible[1].min()
			ymax0 = visible[1].max()

			xmin = np.array([xmin0 - padding, 0]).max()
			xmax = xmax0 + padding
			ymin = np.array([ymin0 - padding, 0]).max()
			ymax = ymax0 + padding

			image_data = image_data[xmin:xmax,ymin:ymax,:]
			image_rgb = image_rgb[xmin0:xmax0,ymin0:ymax0]
			mean = image_rgb.mean()
			# print('min_color: ' + str(min_color) + ' => mean: ' + str(mean))
			min_color = min_color + min_color_step
			if (min_color_stable and min_color_calculated) or mean >= min_avg_color :
				break

		min_color_calculated = True

		# normalising rgb channels
		if normalize_channels:
			maxes = np.array([
				image_data[:,:,0].max(),
				image_data[:,:,1].max(),
				image_data[:,:,2].max()
			])
			image_data = (
				image_data 
				* 
				(
					maxes.max() / maxes
				)
			).round().astype(t)

		if interactive:
			# histogramme
			splt[0].hist(image_data[:,:,0].flatten(), bins=color_channel_values, color='r')
			splt[0].hist(image_data[:,:,1].flatten(), bins=color_channel_values, color='g')
			splt[0].hist(image_data[:,:,2].flatten(), bins=color_channel_values, color='b')
			
			# image finale
			splt[1].imshow(image_data)

			splt[2].set_visible(generate_differential_img and previous_img is not None)
			splt[3].set_visible(generate_differential_img and previous_img is not None)
		
		pil_img = Image.fromarray(image_data)
		file_path = output_dir + '\\' + filename.replace(extension, '.png')
		pil_img.save(file_path)
		print(file_path)

		if generate_differential_img and previous_img is not None:
			#normalize shape with previous image
			xmaxshape = max(image_data.shape[0], previous_img.shape[0])
			ymaxshape = max(image_data.shape[1], previous_img.shape[1])
			maxshape = (xmaxshape,ymaxshape,3)

			image_data_tmp = np.zeros(shape = maxshape)
			image_data_tmp[:image_data.shape[0],:image_data.shape[1],:] = image_data

			previous_img_tmp = np.zeros(shape = maxshape)
			previous_img_tmp[:previous_img.shape[0],:previous_img.shape[1],:] = previous_img
			previous_img = previous_img_tmp
			del previous_img_tmp

			# diff with previous image
			diff_img = image_data_tmp - previous_img
			del image_data_tmp

			tmp_diff_img = np.sum(diff_img, axis=2)
			diff_img = np.zeros(shape = maxshape)

			# red = pixel value decreasing
			# green = pixel value increasing
			# blue = stable pixel
			for x in range(xmaxshape):
				for y in range(ymaxshape):
					val = ((tmp_diff_img[x,y] / 3) / 2) + 127.5
					diff_img[x,y,0] = val if tmp_diff_img[x,y] < 0 else 0
					diff_img[x,y,1] = val if tmp_diff_img[x,y] > 0 else 0
					diff_img[x,y,2] = val if tmp_diff_img[x,y] == 0 else 0
			
			diff_img = diff_img.astype(dtype = t)

			# writing diff image
			pil_img = Image.fromarray(diff_img)
			diff_file_path = output_dir + '\\' + previous_filename.replace(previous_extension, '') + '_TO_' + filename.replace(extension, '.png')
			pil_img.save(diff_file_path)
			print(diff_file_path)

			if interactive:
				# histogramme
				splt[2].hist(diff_img[:,:,0].flatten(), bins=color_channel_values, color='r')
				splt[2].hist(diff_img[:,:,1].flatten(), bins=color_channel_values, color='g')
				splt[2].hist(diff_img[:,:,2].flatten(), bins=color_channel_values, color='b')

				# image finale
				splt[3].imshow(diff_img)

		if generate_differential_img or stack:
			previous_img = image_data
			previous_filename = filename
			previous_extension = extension

		if stack:
			if stack_shape is None:
				stack_shape = image_data.shape
			else:
				stack_shape_x = max(image_data.shape[0], stack_shape[0])
				stack_shape_y = max(image_data.shape[1], stack_shape[1])
				stack_shape = (stack_shape_x,stack_shape_y,3)

			stack_img_tmp = np.zeros(shape = stack_shape)
			if stack_img is not None:
				stack_img_tmp[:stack_img.shape[0],:stack_img.shape[1],:] = stack_img
			stack_img = stack_img_tmp
			del stack_img_tmp

			image_data_tmp = np.zeros(shape = stack_shape)
			image_data_tmp[:image_data.shape[0],:image_data.shape[1],:] = image_data
			stack_img += image_data_tmp
			del image_data_tmp

			nb_stacked_images += 1

		if interactive:
			plt.show(block = True)

if stack:
	stack_img /= nb_stacked_images
	stack_img = stack_img.round().astype(dtype = t)

	# writing stacked image
	pil_img = Image.fromarray(stack_img)
	pil_img.save(stack_file_path)
	print(stack_file_path)

	if interactive:
		# stacked image
		plt.imshow(stack_img)
		plt.show(block = True)

