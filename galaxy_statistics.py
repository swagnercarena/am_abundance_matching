from AbundanceMatching import AbundanceFunction, calc_number_densities, LF_SCATTER_MULT
import numpy as np
from Corrfunc.theory import wp
from matplotlib import pyplot as plt
import matplotlib

# Nice set of colors for plotting
custom_blues = ["#66CCFF", "#33BBFF", "#00AAFF", "#0088CC", "#006699", "#004466"]
custom_blues_complement = ["#FF9966", "#FF7733", "#FF5500", "#CC4400", "#993300",
 "#662200"]

def generate_wp(lf,halos,af_criteria,r_p_data,box_size,mag_cut,pimax=40.0,
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
			mag_cut: The magnitude cut for w_p(r_p)
			pimax: The maximum redshift seperation to use in w_p(r_p) calculation
			nthreads: The number of threads to use for CorrFunc
			scatters: The scatters to deconvolve / re-introduce in the am (must
				be a list)
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
		plt.ylim([0.001,1])
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
		ax[0].set_xlim([np.max(lf[:,0])+2,np.min(lf[:,0])-2])
		ax[0].set_ylim([1e-5,1])
		ax[0].set_ylabel('Number Density (1/ (Mpc^3 h))')
		ax[0].legend(['Fit'] + legend)
		ax[0].set_title('Deconvolved Luminosity Function')
		ax[0].set_yscale('log')
		for r_i in range(len(remainders)):
			ax[1].plot(x, remainders[r_i]/nd,lw=3,
				c=custom_blues_complement[2*r_i])
		ax[1].set_xlabel('Magnitude (M - 5 log h)')
		ax[1].set_ylabel('(LF (deconv) - LF(orig)) / LF(orig)')
		ax[1].set_xlim([np.max(lf[:,0])+2,np.min(lf[:,0])-2])
		plt.show()

	# Conduct the abundance matching
	catalogs = []
	if scatters is not None:
		for scatter in scatters:
			catalogs.append(af.match(nd_halos, scatters[0]*LF_SCATTER_MULT))
	else:
		catalogs = [af.match(nd_halos)]

	wp_binneds = []
	for catalog in catalogs:
		# A luminosity cutoff to use for the correlation function.
		sub_catalog = catalog<mag_cut
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
		wp_binneds.append(wp_binned)

	return wp_binneds




