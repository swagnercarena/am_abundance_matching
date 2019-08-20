from AbundanceMatching import AbundanceFunction, calc_number_densities, LF_SCATTER_MULT
import numpy as np
from Corrfunc.theory import wp
from matplotlib import pyplot as plt
import matplotlib

# Nice set of colors for plotting
custom_blues = ["#66CCFF", "#33BBFF", "#00AAFF", "#0088CC", "#006699", "#004466"]
custom_blues_complement = ["#FF9966", "#FF7733", "#FF5500", "#CC4400", "#993300",
 "#662200"]

def generate_wp(lf,halos,af_criteria,r_p_data,box_size,mag_cuts,pimax=40.0,
	nthreads=1, scatters=None, deconv_repeat = 20, verbose=False):
	"""	Generate the projected 2D correlation by abundance matching galaxies
		Parameters:
			lf: The luminosity function. The first column is the magnitudes and the
				second column is the density in units of 1/Mpc^3.
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

	# Initialize abundance function and calculate the number density of the
	# halos in the box
	af = AbundanceFunction(lf[:,0], lf[:,1], (-25, -5))
	nd_halos = calc_number_densities(halos[af_criteria], 125)
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
			catalogs.append(af.match(nd_halos, scatters[0]*LF_SCATTER_MULT))
	else:
		catalogs = [af.match(nd_halos)]

	wp_binneds = []
	for mag_cut in mag_cuts:
		wp_scatts = []
		for catalog in catalogs:
			# A luminosity cutoff to use for the correlation function.
			sub_catalog = catalog<mag_cut
			print('Scatter %.2f catalog has %d galaxies'%(scatters[len(wp_scatts)],
				np.sum(sub_catalog)))
			x = halos['x'][sub_catalog]
			y = halos['y'][sub_catalog]
			z = halos['z'][sub_catalog]

			# Generate rbins so that the average falls at r_p_data
			rbins = np.zeros(len(r_p_data)+1)
			rbins[1:-1] = 0.5*(r_p_data[:-1]+r_p_data[1:])
			rbins[0] = 2*r_p_data[0]-rbins[1]
			rbins[-1] = 2*r_p_data[-1]-rbins[-2]

			# Calculate the projected correlation function
			wp_results = wp(box_size, pimax, nthreads, rbins, x, y, z, verbose=False, 
				output_rpavg=True)

			# Extract the results
			wp_binned = np.zeros(len(wp_results))
			for i in range(len(wp_results)):
			    wp_binned[i] = wp_results[i][3]
			wp_scatts.append(wp_binned)
		wp_binneds.append(wp_scatts)

	return wp_binneds


def comp_deconv_steps(lf,scatters, deconv_repeats):
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
		figsize=(15,17))
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
			ax[s_i].plot(x, remainder,lw=3,c=custom_blues_complement[2*len(
				legend)])
			y_max = max(y_max,np.max(remainder[x>np.min(lf[:,0])]))
			legend.append('Deconvolution Steps = %d'%(deconv_repeat))

		ax[s_i].set_ylabel('(LF (deconv $\Rightarrow$ conv) - LF) / LF')
		ax[s_i].set_xlim([np.max(lf[:,0])+2,np.min(lf[:,0])])
		ax[s_i].set_title('Luminosity Function Remainder %.2f Scatter'%(
			scatter))



