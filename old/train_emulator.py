import emulator_tools
import glob
import numpy as np

# First load all of the two point correlation functions and their parameters
data_path = '/Users/sebwagner/Documents/Grad_School/Rotations/Risa/abundancematching/wp_results/'
wp_paths = glob.glob(data_path+'wp_*_wp.txt')
p_paths = glob.glob(data_path+'wp_*_p.txt')

print('Loading wprp and associated parameters.')
wp_array = []
p_array = []
for wp_path in wp_paths:
	wp_array.append(np.loadtxt(wp_path))
for p_path in p_paths:
	p_array.append(np.loadtxt(p_path))
wp_array = np.array(wp_array)
# For now drop the second set of points
wp_array = wp_array[:,0,:]
p_array = np.array(p_array)

# This is dirty, but for now just hardcode the rbins
rbins = np.expand_dims(np.array([0.13307,0.22016,0.36166,0.59746,0.98528,1.62909,
	2.68551,4.42953,7.30211,12.0295]),axis=1)

# Transform the data to be emulator friendly
p_array, wp_array = emulator_tools.transform_in_out(p_array,wp_array,rbins)
print(p_array.shape)
print(wp_array.shape)

print('Using parameters to initialize and tune emulator.')
wp_emulator = emulator_tools.initialize_emulator(p_array,wp_array)
emulator_tools.optimize_emulator(wp_emulator,wp_array)
