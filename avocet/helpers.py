import numpy as np
import os

def get_wp_data(wp_data_path,mag_cut,r_cutoff):
    """ Return the two point correlation function.

        Parameters:
            wp_data_path (str): The path to the wp ,dat files
            mag_cut (int): The magnitude cut to load
            r_cutoff (float): The maximum r value to load from the correlation
                function

        Returns:
            (np.array,np.array,np.array): The two point correlation function,
                the covariance matrix, and the variance.
    
    """
    # Load wp
    wp = np.loadtxt(os.path.join(wp_data_path,'wp_%d.dat'%(mag_cut)))
    
    # Load the covariance matrix and deal with the weird formatting
    wp_cov_temp = np.loadtxt(os.path.join(wp_data_path,'wp_covar_%d.dat'%(mag_cut)))
    wp_cov = np.zeros((len(wp),len(wp)))
    for wp_tup in wp_cov_temp:
        wp_cov[int(wp_tup[0])-1,int(wp_tup[1])-1] = wp_tup[2]
        wp_cov[int(wp_tup[1])-1,int(wp_tup[0])-1] = wp_tup[2]
    
    # Apply the cutoff on radius
    wp_cov=wp_cov[wp[:,0]<r_cutoff,:]
    wp_cov=wp_cov[:,wp[:,0]<r_cutoff]
    wp_var = np.diag(wp_cov)
    wp = wp[wp[:,0]<r_cutoff]
    
    return wp, wp_cov, wp_var