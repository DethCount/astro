# pip install matplotlib astropy
# python planetary_crop.py

import os
import matplotlib.pyplot as plt
from astropy.visualization import astropy_mpl_style
plt.style.use(astropy_mpl_style)

from astropy.utils.data import get_pkg_data_filename
from astropy.io import fits

import numpy as np
from PIL import Image

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

min_color_calculated = False 
color_channel_values = range(255);
previous_img = None
previous_filename = None

for filename in os.listdir(input_dir):
	if filename.endswith(".fit") or filename.endswith(".fits"):
		extension = 'fit' if filename.endswith(".fit") else 'fits'

		image_file = get_pkg_data_filename(input_dir + '\\' + filename)
		fits.info(image_file)
		image_data = fits.getdata(image_file, ext=0)
		image_data = image_data.transpose((1,2,0))
		image_data = np.flip(image_data, axis=0)
		t = image_data.dtype

		if interactive:
			fig,splt = plt.subplots(4 if generate_differential_img else 2)

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
		file_path = output_dir + '\\' + filename.replace('.'+extension, '') + '.png'
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
			diff_file_path = output_dir + '\\' + previous_filename.replace('.'+extension, '') + '_TO_' + filename.replace('.'+extension, '') + '.png'
			pil_img.save(diff_file_path)
			print(diff_file_path)

			if interactive:
				# histogramme
				splt[2].hist(diff_img[:,:,0].flatten(), bins=color_channel_values, color='r')
				splt[2].hist(diff_img[:,:,1].flatten(), bins=color_channel_values, color='g')
				splt[2].hist(diff_img[:,:,2].flatten(), bins=color_channel_values, color='b')

				# image finale
				splt[3].imshow(diff_img)

		if generate_differential_img:
			previous_img = image_data
			previous_filename = filename

		if interactive:
			plt.show(block = True)
