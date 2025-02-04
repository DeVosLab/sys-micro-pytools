import numpy as np

def unsqueeze_to_ndim(img, n_dim):
	'''Unsqueeze an image to a given number of dimensions.

	Parameters
	----------
	img : ndarray
		Image to unsqueeze.
	n_dim : int
		Target number of dimensions.
	
	Returns
	-------
	img : ndarray
		Unsqueezed image with n_dim dimensions.	
	'''

	if img.ndim <= n_dim:
		raise ValueError(f'img.ndim should be greater than n_dim, but found img.ndim={img.ndim} and n_dim={n_dim}')
	if img.ndim < n_dim:
		img = np.expand_dims(img,0)
		img = unsqueeze_to_ndim(img, n_dim)
	return img


def squeeze_to_ndim(img, n_dim):
	'''Squeeze an image to a given number of dimensions.

	Parameters
	----------
	img : ndarray
		Image to squeeze.
	n_dim : int
		Target number of dimensions.

	Returns
	-------
	img : ndarray
		Squeezed image with n_dim dimensions.
	'''

	if img.ndim > n_dim:
		if img.ndim >= n_dim:
			raise ValueError(f'img.ndim should be greater than or equal to n_dim, but found img.ndim={img.ndim} and n_dim={n_dim}')
		if not any([s == 1 for s in img.shape]):
			raise ValueError('img cannot be further squeezed, as it does not contain any singleton dimensions')
		img = np.squeeze(img)
		img = squeeze_to_ndim(img, n_dim)
	
	return img