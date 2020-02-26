from AbundanceMatching import *
import emcee, pandas, time
from os.path import exists
import numpy as np
import Corrfunc
from Corrfunc.theory import wp
from astropy.io import fits
from galaxy_statistics import AMLikelihood
from tqdm import tqdm
from multiprocessing import Pool

# First load all of the data we'll use for the MCMC sampling

data_path = '/u/ki/rmredd/data/'
# Luminosity function
# lf_21 = np.loadtxt(data_path + 'lf/tinker/lf_jt_21.dat')
lf_20 = np.loadtxt(data_path + 'lf/tinker/lf_jt_20.dat')
lf_18 = np.loadtxt(data_path + 'lf/tinker/lf_jt_18.dat')
# lf_21 = lf_21[lf_21[:,1]>0,:]
lf_20 = lf_20[lf_20[:,1]>0,:]
lf_18 = lf_18[lf_18[:,1]>0,:]

wp_path = '/u/ki/rmredd/data/corr_wp/tinker_sdss_wp/'
wp_20 = np.loadtxt(wp_path + 'wp_20.dat')
wp_20_cov_temp = np.loadtxt(wp_path + 'wp_covar_20.dat')
wp_20_cov = np.zeros((len(wp_20),len(wp_20)))
for wp_tup in wp_20_cov_temp:
	wp_20_cov[int(wp_tup[0])-1,int(wp_tup[1])-1] = wp_tup[2]
	wp_20_cov[int(wp_tup[1])-1,int(wp_tup[0])-1] = wp_tup[2]
	
wp_18 = np.loadtxt(wp_path + 'wp_18.dat')
wp_18_cov_temp = np.loadtxt(wp_path + 'wp_covar_18.dat')
wp_18_cov = np.zeros((len(wp_18),len(wp_18)))
for wp_tup in wp_18_cov_temp:
	wp_18_cov[int(wp_tup[0])-1,int(wp_tup[1])-1] = wp_tup[2]
	wp_18_cov[int(wp_tup[1])-1,int(wp_tup[0])-1] = wp_tup[2]
	
wp_21 = np.loadtxt(wp_path + 'wp_21.dat')
wp_21_cov_temp = np.loadtxt(wp_path + 'wp_covar_21.dat')
wp_21_cov = np.zeros((len(wp_21),len(wp_21)))
for wp_tup in wp_21_cov_temp:
	wp_21_cov[int(wp_tup[0])-1,int(wp_tup[1])-1] = wp_tup[2]
	wp_21_cov[int(wp_tup[1])-1,int(wp_tup[0])-1] = wp_tup[2]
	
r_cutoff = 15

wp_20_cov=wp_20_cov[wp_20[:,0]<r_cutoff,:]
wp_20_cov=wp_20_cov[:,wp_20[:,0]<r_cutoff]
wp_20_var = np.diag(wp_20_cov)
wp_20 = wp_20[wp_20[:,0]<r_cutoff]

wp_18_cov=wp_18_cov[wp_18[:,0]<r_cutoff,:]
wp_18_cov=wp_18_cov[:,wp_18[:,0]<r_cutoff]
wp_18_var = np.diag(wp_18_cov)
wp_18 = wp_18[wp_18[:,0]<r_cutoff]

wp_21_cov=wp_21_cov[wp_21[:,0]<r_cutoff,:]
wp_21_cov=wp_21_cov[:,wp_21[:,0]<r_cutoff]
wp_21_var = np.diag(wp_21_cov)
wp_21 = wp_21[wp_21[:,0]<r_cutoff]

box_size = 400
pimax = 40.0
deconv_repeat = 200
mag_cuts=[-21.0,-20.0]
# Use lf_20 twice since we don't have an lf_21
lf_list = [lf_20,lf_20]

# Halos from n body sim
halo_path = '/nfs/slac/des/fs1/g/sims/jderose/BCCSims/c400-2048/'
halos = np.array(fits.open(halo_path + 'hlist_1.00000.list.fits')[1].data)
r_p_data = wp_20[:,0]
wp_data_list = [wp_21[:,1],wp_20[:,1]]
wp_cov_list = [wp_21_cov,wp_20_cov]
nthreads = 1

wp_save_path = '/u/ki/swagnerc/abundance_matching/wp_results/wp'

af_criteria = 'vmax'

like_class = AMLikelihood(lf_list,halos,af_criteria,box_size,r_p_data,mag_cuts,
	wp_data_list,wp_cov_list,pimax,nthreads,deconv_repeat,wp_save_path)

# Test speed of a single function call
like_class.wp_save_path = '/u/ki/swagnerc/abundance_matching/wp_results/timing'
start = time.time()
like_class.log_likelihood([0.1,0.1])
print('One evaluation takes %.2f seconds'%(time.time()-start))
like_class.wp_save_path = wp_save_path

n_params = 2; n_walkers = 10;
n_steps = 1000
pos = np.random.rand(n_params*n_walkers).reshape((n_walkers,n_params))*0.3

import csv   
fields=['scatter','mu_cut']
csv_path = '/u/ki/swagnerc/abundance_matching/wp_results/mc_chains.csv'
if exists(csv_path):
	# Load up most recent set of positions
	frame = pandas.read_csv(csv_path)
	pos = frame.values[-n_walkers:]

print(pos)
sampler = emcee.EnsembleSampler(n_walkers, n_params, like_class.log_likelihood)
with open(csv_path, 'a',1) as f:
	writer = csv.writer(f)
	if not exists(csv_path):
		writer.writerow(fields)
	save_step = 1
	for step in tqdm(range(n_steps//save_step+1)):
		pos, _, _ = sampler.run_mcmc(pos, save_step)
		writer.writerows(sampler.chain[:,-save_step:,:].reshape(-1,n_params))
