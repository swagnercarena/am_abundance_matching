import george
from george.kernels import ExpSquaredKernel
import numpy as np
import scipy.optimize as op

def transform_in_out(am_param_train,wprp_train,rbins):
	"""
	Transform input and output space such that the inputs are abundance matching
	parameters and a radial bin, and the output is the value of wprp for that 
	combination of parameters and bin.

	Parameters:
		am_param_train: A numpy array containing the abundance matching 
			parameters for each training point. This should have dimensions
			(n_training points x nun_params)
		wprp_train: The projected two point correlation function for each 
			set of abundance matching parameters
		rbins: The median value of the radial bins

	Returns:
		am_param_train,wprp_train with the radial bins one of the input 
			parameters.
	"""
	# All we have to do to wprp is flatten it.
	wprp_train = wprp_train.flatten()
	# Initialize the shape of the new training inputs
	am_param_train_rbins = np.zeros((am_param_train.shape[0]*len(rbins),
		am_param_train.shape[1]+1))
	for ampi in range(len(am_param_train)):
		for rbinsi in range(len(rbins)):
			am_param_train_rbins[ampi*len(rbins)+rbinsi] = np.concatenate(
				[am_param_train[ampi],rbins[rbinsi]],axis=0)
	return am_param_train_rbins,wprp_train


def initialize_emulator(am_param_train,wprp_train):
	"""
	Given a set of abundance matching parameters and projectedtwo point 
	correlation functions to use for training, build an emulator.

	Parameters:
		am_param_train: A numpy array containing the abundance matching 
			parameters for each training point. This should have dimensions
			(n_training points x nun_params)
		wprp_train: The projected two point correlation function for each 
			set of abundance matching parameters

	Returns:
		An emulator initialized to the training points provided
	"""
	# Get the number of parameters for your abundance matching model
	n_am_params = am_param_train.shape[1]
	# Randomly initialzie our emulator parameters. The exact number of
	# parameters required depends on the kernel.
	em_vec = np.random.rand(n_am_params+2)

	sf = em_vec[0]
	sx = em_vec[-1]
	# This kernel was suggested by sean. Will likely have to experiment with
	# different kernel variaties
	kernel = sf * ExpSquaredKernel(em_vec[1:n_am_params+1], 
		ndim=n_am_params) + sx
	emulator = george.GP(kernel, mean=np.mean(wprp_train))
	emulator.compute(am_param_train)
	return emulator

def optimize_emulator(emulator,wprp_train):
	"""
	Find the local minimum of the emulator hyperparameters.

	Parameters:
		emulator: The emulator object (from George). Must have been initialized
		am_param_train: The am parameters to use to train the hyperparameter
			values
		wprp_train: The projected two point correlation functions for each of
			the abundance matching configurations

	Returns:
		None. The emulator object will be updated with the optimal
			hyperparameter values.
	"""
	def nll(vector):
		# Update the kernel params and calculate nll
		emulator.kernel.set_parameter_vector(vector)
		ll = emulator.lnlikelihood(wprp_train, quiet=True)
		# Deal with scipy not liking infinities as per george example
		return -ll if np.isfinite(ll) else 1e25

	def grad_nll(vector):
		# Again update the kernel and calculate the grad of nll with respect
		# to the vector.
		emulator.kernel.set_parameter_vector(vector)
		return -emulator.grad_lnlikelihood(wprp_train, quiet=True)

	# To start, run the standard scipy optimizer on our emulator
	vec0 = emulator.kernel.parameter_vector
	results = op.minimize(nll,vec0,jac=grad_nll, method="L-BFGS-B")

	# Update our kernel with the final step
	emulator.kernel.set_parameter_vector(results.x)


