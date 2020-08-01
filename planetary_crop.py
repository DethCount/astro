import os
import matplotlib.pyplot as plt
from astropy.visualization import astropy_mpl_style
plt.style.use(astropy_mpl_style)

from astropy.utils.data import get_pkg_data_filename
from astropy.io import fits

import numpy as np
from PIL import Image

input_dir = 'in'
output_dir = 'out'
interactive = False
normalize_channels = False
min_avg_color = 0.6
min_color_stable = False
min_color_init = 10
min_color_step = 5
min_color_calculated = False
padding = 10

color_channel_values = range(255);

for filename in os.listdir(input_dir):
	if filename.endswith(".fit") or filename.endswith(".fits"):
		extension = 'fit' if filename.endswith(".fit") else 'fits'

		image_file = get_pkg_data_filename(input_dir + '\\' + filename)
		fits.info(image_file)
		image_data = fits.getdata(image_file, ext=0)
		image_data = image_data.transpose((1,2,0))

		min_color = min_color if min_color_calculated and min_color_stable else min_color_init

		# filtering dark pixels
		while True:
			image_rgb = image_data
			image_rgb = np.sum(image_rgb, axis=2)
			image_rgb[image_rgb < min_color] = 0
			image_rgb[image_rgb >= min_color] = 1

			# croping image
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
			print('min_color: ' + str(min_color) + ' => mean: ' + str(mean))
			min_color = min_color + min_color_step
			if (min_color_stable and min_color_calculated) or mean >= min_avg_color :
				break

		min_color_calculated = True

		# normalising rgb channels
		if normalize_channels:
			t = image_data.dtype
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
			plt.hist(image_data[:,:,0].flatten(), bins=color_channel_values, color='r')
			plt.hist(image_data[:,:,1].flatten(), bins=color_channel_values, color='g')
			plt.hist(image_data[:,:,2].flatten(), bins=color_channel_values, color='b')
			plt.show(block = False)

			# image finale
			plt.figure()
			plt.imshow(image_data)
			plt.colorbar()
			plt.show()

		image_data = np.flip(image_data, axis=0)
		pil_img = Image.fromarray(image_data)
		pil_img.save(output_dir + '\\' + filename.replace('.'+extension, '') + '.png')
		print(output_dir + '\\' + filename.replace('.'+extension, '') + '.png')