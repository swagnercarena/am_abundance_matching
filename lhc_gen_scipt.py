import argparse
import emulator_tools
from galaxy_statistics import AMLikelihood
import numpy as np
from astropy.io import fits

def main():
	# Get the basic arguments about what kind of lhc we want to generate
	parser = argparse.ArgumentParser()
	parser.add_argument('n_points',help='The number of lhc points to make')
	parser.add_argument('n_wp_samps',help='The number of wprp samples to use '+
		'to estimate the variance in wprp calculation.')
	parser.add_argument('dict_path',help='The path to save the dictionary '+
		'to')
	args = parser.parse_args()

	n_points = int(args.n_points)
	n_wp_samps = int(args.n_wp_samps)

	# These are set by hand in the config file for now. Sorry!
	params_min = [0.001,0.001]
	params_max = [0.35,0.35]

	# Load all the data
	print('Loading the data.')
	data_path = '/u/ki/rmredd/data/'
	# Luminosity function
	# lf_21 = np.loadtxt(data_path + 'lf/tinker/lf_jt_21.dat')
	lf_20 = np.loadtxt(data_path + 'lf/tinker/lf_jt_20.dat')
	# lf_21 = lf_21[lf_21[:,1]>0,:]
	lf_20 = lf_20[lf_20[:,1]>0,:]

	wp_path = '/u/ki/rmredd/data/corr_wp/tinker_sdss_wp/'
	wp_20 = np.loadtxt(wp_path + 'wp_20.dat')
	wp_20_cov_temp = np.loadtxt(wp_path + 'wp_covar_20.dat')
	wp_20_cov = np.zeros((len(wp_20),len(wp_20)))
	for wp_tup in wp_20_cov_temp:
		wp_20_cov[int(wp_tup[0])-1,int(wp_tup[1])-1] = wp_tup[2]
		wp_20_cov[int(wp_tup[1])-1,int(wp_tup[0])-1] = wp_tup[2]
		
	wp_21 = np.loadtxt(wp_path + 'wp_21.dat')
	wp_21_cov_temp = np.loadtxt(wp_path + 'wp_covar_21.dat')
	wp_21_cov = np.zeros((len(wp_21),len(wp_21)))
	for wp_tup in wp_21_cov_temp:
		wp_21_cov[int(wp_tup[0])-1,int(wp_tup[1])-1] = wp_tup[2]
		wp_21_cov[int(wp_tup[1])-1,int(wp_tup[0])-1] = wp_tup[2]
		
	r_cutoff = 15

	wp_20_cov=wp_20_cov[wp_20[:,0]<r_cutoff,:]
	wp_20_cov=wp_20_cov[:,wp_20[:,0]<r_cutoff]
	wp_20 = wp_20[wp_20[:,0]<r_cutoff]

	wp_21_cov=wp_21_cov[wp_21[:,0]<r_cutoff,:]
	wp_21_cov=wp_21_cov[:,wp_21[:,0]<r_cutoff]
	wp_21 = wp_21[wp_21[:,0]<r_cutoff]

	box_size = 400
	pimax = 40.0
	deconv_repeat = 20
	mag_cuts=[-21.0,-20.0]

	# Halos from n body sim
	halo_path = '/nfs/slac/des/fs1/g/sims/jderose/BCCSims/c400-2048/'
	halos = np.array(fits.open(halo_path + 'hlist_1.00000.list.fits')[1].data)
	r_p_data = wp_20[:,0]
	nthreads = 1

	lf_list = [lf_20,lf_20]
	wp_data_list = [wp_21[:,1],wp_20[:,1]]
	wp_cov_list = [wp_21_cov,wp_20_cov]
	wp_save_path = '/u/ki/swagnerc/abundance_matching/wp_results/emu_test'
	print('Creating likelihood class to compute wprp')
	like_class = AMLikelihood(lf_list,halos,'vmax',box_size,r_p_data,mag_cuts,
		wp_data_list,wp_cov_list,pimax,nthreads,deconv_repeat,wp_save_path)


	print('Generating Dictionary')
	wp_train_dict = emulator_tools.generate_lhc(like_class,
		n_points,params_min,params_max,n_wp_samps)
	np.save(args.dict_path,wp_train_dict)

if __name__ == '__main__':
    main()


