from AbundanceMatching import AbundanceFunction, calc_number_densities, LF_SCATTER_MULT
import numpy as np
from Corrfunc.theory import wp
from matplotlib import pyplot as plt
from scipy.spatial import cKDTree
import matplotlib

# Nice set of colors for plotting
custom_blues = ["#66CCFF", "#33BBFF", "#00AAFF", "#0088CC", "#006699", "#004466"]
custom_blues_complement = ["#FF9966", "#FF7733", "#FF5500", "#CC4400", "#993300",
 "#662200"]

def generate_wp(lf_list,halos,af_criteria,r_p_data,box_size,mag_cuts,pimax=40.0,
	nthreads=1, scatters=None, deconv_repeat = 20, verbose=False):
	"""	Generate the projected 2D correlation by abundance matching galaxies
		Parameters:
			lf_list: A list of luminosity functions for each mag_cut. The first 
				column is the magnitudes and thesecond column is the density in 
				units of 1/Mpc^3.
			halos: A catalog of the halos in the n-body sim that can be indexed
				into using the quantity name.
			af_criteria: The galaxy property (i.e. vpeak) to use for abundance 
				matching.
			r_p_data: The positions at which to calculate the 2D correlation
				function.
			box_size: The size of the box (box length not volume)
			mag_cuts: The magnitude cuts for w_p(r_p) (must be a list)
			pimax: The maximum redshift seperation to use in w_p(r_p) calculation
			nthreads: The number of threads to use for CorrFunc
			scatters: The scatters to deconvolve / re-introduce in the am (must
				be a list)
			deconv_repeat: The number of deconvolution steps to conduct
			verbose: If set to true, will generate plots for visual inspection
				of am outputs.
		Returns:
			w_p(r_p) at the r_p values specified by r_p_data.
	"""
	# Repeat once for each magnitude cut
	wp_binneds = []
	for mag_cut_i in range(len(mag_cuts)):
		mag_cut = mag_cuts[mag_cut_i]
		lf = lf_list[mag_cut_i]
		
		# Initialize abundance function and calculate the number density of the
		# halos in the box
		af = AbundanceFunction(lf[:,0], lf[:,1], (-25, -5))
		nd_halos = calc_number_densities(halos[af_criteria], box_size)
		if scatters is not None:
			remainders = []
			for scatter in scatters:
				remainders.append(
					af.deconvolute(scatter*LF_SCATTER_MULT, deconv_repeat))

		# If verbose output the match between abundance function and input data
		if verbose:
			matplotlib.rcParams.update({'font.size': 18})
			plt.figure(figsize=(10,8))
			plt.plot(lf[:,0], lf[:,1],lw=7,c=custom_blues[1])
			x = np.linspace(np.min(lf[:,0])-2, np.max(lf[:,0])+2, 101)
			plt.semilogy(x, af(x),lw=3,c=custom_blues[4])
			plt.xlim([np.max(lf[:,0])+2,np.min(lf[:,0])])
			plt.ylim([1e-5,1])
			plt.xlabel('Magnitude (M - 5 log h)')
			plt.ylabel('Number Density (1/ (Mpc^3 h))')
			plt.legend(['Input','Fit'])
			plt.title('Luminosity Function')
			plt.yscale('log')
			plt.show()

		# Plot remainder to ensure the deconvolution returned reasonable results
		if verbose and scatters is not None:
			f, ax = plt.subplots(2,1, sharex='col', sharey='row', figsize=(15,12), 
				gridspec_kw={'height_ratios':[2, 1]})

			x, nd = af.get_number_density_table()
			ax[0].plot(x, nd,lw=3,c=custom_blues[4])
			legend = []
			for scatter in scatters:
				ax[0].plot(af._x_deconv[float(scatter*LF_SCATTER_MULT)],nd,lw=3,
					c=custom_blues_complement[2*len(legend)])
				legend.append('Scatter = %.2f'%(scatter))
			ax[0].set_xlim([np.max(lf[:,0])+2,np.min(lf[:,0])])
			ax[0].set_ylim([1e-5,1])
			ax[0].set_ylabel('Number Density (1/ (Mpc^3 h))')
			ax[0].legend(['Fit'] + legend)
			ax[0].set_title('Deconvolved Luminosity Function')
			ax[0].set_yscale('log')
			ax[1].set_xlabel('Magnitude (M - 5 log h)')
			ax[1].set_ylabel('(LF (deconv $\Rightarrow$ conv) - LF) / LF')
			ax[1].set_xlim([np.max(lf[:,0])+2,np.min(lf[:,0])])
			y_max = 0
			for r_i in range(len(remainders)):
				remainder = remainders[r_i]/nd
				ax[1].plot(x, remainder,lw=3,
					c=custom_blues_complement[2*r_i])
				y_max = max(y_max,np.max(remainder[x>np.min(lf[:,0])]))
			ax[1].set_ylim([-1.2,y_max*1.2])
			plt.show()

		# Conduct the abundance matching
		catalogs = []
		if scatters is not None:
			for scatter in scatters:
				catalogs.append(af.match(nd_halos, scatter*LF_SCATTER_MULT,
					do_rematch=False))
		else:
			catalogs = [af.match(nd_halos)]

		wp_scatts = []
		for catalog in catalogs:
			# A luminosity cutoff to use for the correlation function.
			sub_catalog = catalog<mag_cut
			print('Scatter %.2f catalog has %d galaxies'%(scatters[len(wp_scatts)],
				np.sum(sub_catalog)))
			x = halos['px'][sub_catalog]
			y = halos['py'][sub_catalog]
			z = halos['pz'][sub_catalog]

			# Generate rbins so that the average falls at r_p_data
			rbins = np.zeros(len(r_p_data)+1)
			rbins[1:-1] = 0.5*(r_p_data[:-1]+r_p_data[1:])
			rbins[0] = 2*r_p_data[0]-rbins[1]
			rbins[-1] = 2*r_p_data[-1]-rbins[-2]

			# Calculate the projected correlation function
			wp_results = wp(box_size, pimax, nthreads, rbins, x, y, z, 
				verbose=False, output_rpavg=True)

			# Extract the results
			wp_binned = np.zeros(len(wp_results))
			for i in range(len(wp_results)):
			    wp_binned[i] = wp_results[i][3]
			wp_scatts.append(wp_binned)
		wp_binneds.append(wp_scatts)

	return wp_binneds


def comp_deconv_steps(lf,scatters, deconv_repeats,m_max=-25):
	"""	Generate the projected 2D correlation by abundance matching galaxies
		Parameters:
			lf: The luminosity function. The first column is the magnitudes and the
				second column is the density in units of 1/Mpc^3.
			scatters: The scatters to deconvolve / re-introduce in the am (must
				be a list)
		Returns:
			w_p(r_p) at the r_p values specified by r_p_data.
	"""

	# Initialize abundance function and calculate the number density of the
	# halos in the box
	af = AbundanceFunction(lf[:,0], lf[:,1], (-25, -5))

	f, ax = plt.subplots(len(scatters),1, sharex='col', sharey='row', 
		figsize=(13,16))
	ax[-1].set_xlabel('Magnitude (M - 5 log h)')
	x, nd = af.get_number_density_table()

	# For each scatter and each number of deconvolution steps, plot the
	# dependence of the remainder on the step.
	for s_i in range(len(scatters)):
		scatter = scatters[s_i]
		y_max = 0
		legend = []

		for deconv_repeat in deconv_repeats:
			remainder = af.deconvolute(scatter*LF_SCATTER_MULT, deconv_repeat)/nd
			ax[s_i].plot(x, remainder,lw=3,c=custom_blues_complement[len(
				legend)])
			y_max = max(y_max,np.max(remainder[x>m_max]))
			legend.append('Deconvolution Steps = %d'%(deconv_repeat))

		ax[s_i].set_ylabel('(LF (deconv $\Rightarrow$ conv) - LF) / LF')
		ax[s_i].set_xlim([np.max(lf[:,0])-2,m_max])
		ax[s_i].set_ylim([-1.2,y_max*1.2])
		ax[s_i].set_title('Luminosity Function Remainder %.2f Scatter'%(
			scatter))

	ax[0].legend(legend)

class AMLikelihood(object):
	""" A class responsible for AM likelihood calculations for parameter fitting.
		Currently assumes a fixed cosmology.
	"""
	def __init__(self,lf_list,halos,af_criteria,box_size,r_p_data,mag_cuts,
		wp_data_list, wp_cov_list, pimax, nthreads, deconv_repeat,
		wp_save_path,n_k_tree_cut = None):
		""" Initialize AMLikelihood object. This involves initializing an
			AbundanceFunction object for each luminosity function.
			Parameters:
				lf_list: List of luminosity functions
				halos: Halos dictionairy. halos[af_criteria] should return
					a numpy array of values for the abundance matching
					criteria for the halos.
				af_criteria: A string that will be used to index into halos
					for the abundance matching criteria data.
				box_size: The size of the box being used (length)
				r_p_data: The positions at which to calculate the 2D correlation
					function.
				mag_cuts: The magnitude cuts for w_p(r_p) (must be a list)
				wp_data_list: The list of wp_data corresponding to lf_list 
					to be used for the likelihood function
				wp_cov_list: The list of covariance matrices for w_p. Also
					important for the covariance function.
				pimax: The maximum redshift seperation to use in w_p(r_p) 
					calculation
				nthreads: The number of threads to use for CorrFunc
				deconv_repeat: The number of deconvolution steps to conduct
				wp_save_path: A unique path to which to save the parameter
					values and 2d projected correlation functions.
				n_k_tree_cut: An integer for the number of halos to cut from
					the catalog in the k nearest neighbor step. This will reduce
					accuracy at small scales but speed up computation. If set
					to None this step will not be done.
			Output:
				Initialized class
		"""
		# Save dictionairy parameter along with box size object
		# pre-sort halos to speed up computations
		self.halos = halos[np.argsort(halos[af_criteria])]
		self.af_criteria = af_criteria
		self.box_size = box_size
		self.r_p_data = r_p_data
		self.mag_cuts = mag_cuts
		self.wp_data_list = wp_data_list
		self.wp_cov_list = wp_cov_list
		self.pimax = pimax
		self.nthreads = nthreads
		self.deconv_repeat = deconv_repeat
		# Generate list of abundance matching functions
		self.af_list = []
		for lf in lf_list:
			af = AbundanceFunction(lf[:,0], lf[:,1], (-25, -5))
			self.af_list.append(af)

		# Generate rbins so that the average falls at r_p_data
		rbins = np.zeros(len(r_p_data)+1)
		rbins[1:-1] = 0.5*(r_p_data[:-1]+r_p_data[1:])
		rbins[0] = 2*r_p_data[0]-rbins[1]
		rbins[-1] = 2*r_p_data[-1]-rbins[-2]
		self.rbins = rbins
		self.wp_save_path = wp_save_path

		# K nearest neighbors calculation for the cut. This only needs
		# to be done once since the cut is not dependent on the AM 
		# parameters.
		if n_k_tree_cut is not None:
			neigh_pos = np.transpose(np.vstack([
				self.halos['px'],self.halos['py']]))
			# Epsilon in case some galaxies are cataloges as being at the edge
			# of the box.
			epsilon = 1e-12
			# Set up the tree
			tree = cKDTree(neigh_pos,boxsize=box_size+epsilon)
			# Query the 2nd nearest neighbor.
			dist, locs = tree.query(neigh_pos,k=2)
			keep = np.argsort(dist[:,1])[n_k_tree_cut:]
			# A bool array to use for indexing
			self.wp_keep = np.zeros(len(halos),dtype=bool)
			self.wp_keep[keep] = True

		else:
			self.wp_keep = None

	def log_likelihood(self, params, verbose=False):
		""" Calculate the loglikelihood of the particular parameter values
			for abundance matching given the data. Currently supports
			scatter and mu_cut.
			Parameters:
				params: A vector containing [scatter,mu_cut] to be tested.
			Output:
				The log likelihood.
		"""
		scatter = params[0]
		mu_cut  = params[1]
		if scatter < 0.0 or mu_cut < 0.0:
			return -np.inf
		# We assume here that the maximum mass is stored as mvir and 
		# the current mass is stored as mvir_now. Need to be changed if the
		# dictionairy changes (or made more general).
		halos_post_cut = self.halos['mvir_now']/self.halos['mvir'] > mu_cut

		# Calculate what to remove due to k_nearest_neighbors
		if self.wp_keep is not None:
			wp_post_cut_keep = self.wp_keep[halos_post_cut]
		else:
			wp_post_cut_keep = np.ones(np.sum(halos_post_cut),dtype=bool)

		nd_halos = calc_number_densities(self.halos[self.af_criteria][
			halos_post_cut], self.box_size)
		# Deconvolve the scatter and generate catalogs for each mag_cut
		catalog_list = []
		for af in self.af_list:
			af.deconvolute(scatter*LF_SCATTER_MULT,self.deconv_repeat)
			catalog_list.append(af.match(nd_halos,scatter*LF_SCATTER_MULT,
				do_rematch=False))

		log_like = 0
		wp_saved_results = []
		for c_i in range(len(catalog_list)):
			catalog = catalog_list[c_i]
			sub_catalog = catalog[wp_post_cut_keep] < self.mag_cuts[c_i]

			# Extract positions of halos in our catalog
			x = self.halos['px'][halos_post_cut]; x=x[wp_post_cut_keep]
			x=x[sub_catalog]
			y = self.halos['py'][halos_post_cut]; y=y[wp_post_cut_keep]
			y=y[sub_catalog]
			z = self.halos['pz'][halos_post_cut]; z=z[wp_post_cut_keep]
			z=z[sub_catalog]

			# Get the wp for the catalog
			wp_results = wp(self.box_size, self.pimax, self.nthreads, 
				self.rbins, x, y, z, verbose=verbose, 
				output_rpavg=True)
			wp_binned = np.zeros(len(wp_results))
			for i in range(len(wp_results)):
			    wp_binned[i] = wp_results[i][3]
			wp_saved_results.append(wp_binned)

			dif_vector = wp_binned - self.wp_data_list[c_i]
			log_like += - 0.5*np.dot(np.dot(dif_vector,np.linalg.inv(
				self.wp_cov_list[c_i])),dif_vector)

		wp_saved_results = np.array(wp_saved_results)
		np.savetxt(self.wp_save_path+'_%d%d_wp.txt'%(scatter*1e6,mu_cut*1e6),
			wp_saved_results)
		np.savetxt(self.wp_save_path+'_%d%d_p.txt'%(scatter*1e6,mu_cut*1e6),
			params)

		return log_like






