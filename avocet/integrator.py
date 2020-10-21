import numba
import numpy as np

# Declare some global variables
G = 6.67430e-11


@numba.njit()
def leapfrog_p_step(x0,v0,dt,a0):
	"""	Given the current position, velocity, and acceleration of a particle
		conduct one position step of leapfrog integration.

		Parameters:
			x0 (np.array): The current 3d position. Modified in place to x1
			v0 (np.array): The current 3d velocity. Modified in place to vp
			dt (float): The timestep for integration
			a0 (np.array): The 3d gradient of the potential before the current
				leapfrog step.

	"""
	x0 += v0*dt + 0.5*a0*dt**2


@numba.njit()
def leapfrog_v_step(v0,dt,a0,a1):
	"""	Given the current position, velocity, and acceleration of a particle
		conduct one velocity step of leapfrog integration.

		Parameters:
			vp (np.array): The current 3d velocity. Modified in place to v1
			dt (float): The timestep for integration
			a0 (np.array): The 3d gradient of the potential before the current
				leapfrog step.
			a0 (np.array): The 3d gradient of the potential after the current
				leapfrog step.

	"""
	v0 += 0.5*(a1+a0)*dt


"""----------------------------N Body Functions---------------------------"""


@numba.njit()
def calc_neg_grad_nb(part_pos,part_mass,pos):
	""" Calculate the negative gradient of the n-body potential at a specific
		position.

		Parameters:
			part_pos (np.array): The Nx3 particle positions
			part_mass (np.array): The N particle masses
			pos (np.array): The 3D position to calculate the gradient at

		Returns:
			(np.array): The 3D negative gradient of the potential
	"""
	grad = np.zeros(3,dtype=np.float64)
	# Add the contribution from each point mass to the potential
	for pi in range(len(part_pos)):
		# Skip massless particles. This will mostly be used to ignore
		# the particle itself in the potential.
		if part_mass[pi] == 0:
			continue
		# Get the difference in position
		pos_dif = pos-part_pos[pi]
		pos_dif_2 = np.sum(np.square(pos_dif))
		grad += -G*part_mass[pi]/np.power(pos_dif_2,1.5)*pos_dif

	return grad


@numba.njit()
def leapfrog_int_nb(part_pos,part_vel,part_mass,dt,num_dt,save_pos_array,
	save_vel_array):
	"""	Integrate the n body problem

		Parameters:
			part_pos (np.array): The Nx3D particle positions. This will be
				modified in place.
			part_vel (np.array): The Nx3D particle velocities. This will be
				modified in place.
			part_mass (np.array): The N particle masses.
			dt (float): The size of a timestep
			num_dt (int): The number of steps to take
			save_pos_array (np.array): A (num_dt+1)*N*3D array where the
				positions will be saved.
			save_vel_array (np.array): A (num_dt+1)*N*3D array where the
				velocities will be saved.
	"""
	# Number of particles
	num_p = len(part_pos)

	# We need to store the current potential for the
	# particles before conducting the update
	a0_vec = np.zeros((num_p,3),dtype=np.float64)

	# Variables we'll use for inplace calculation
	save_mass = 0

	# Clear save arrays in case they have values
	save_pos_array *= 0
	save_vel_array *= 0

	for ti in range(num_dt):
		# Save the pos and vel at this step
		save_pos_array[ti] += part_pos
		save_vel_array[ti] += part_vel
		# First get the a0 for each particle
		for pi in range(num_p):
			# Get the position and velocity of our particle
			pos = part_pos[pi]

			# Set the mass of this particle to 0
			save_mass = part_mass[pi]
			part_mass[pi] = 0

			a0_vec[pi] = calc_neg_grad_nb(part_pos,part_mass,pos)

			# Reset the mass
			part_mass[pi] = save_mass

		for pi in range(num_p):
			# Update the position of our particles
			pos = part_pos[pi]
			vel = part_vel[pi]

			leapfrog_p_step(pos,vel,dt,a0_vec[pi])

		# Now do the leapfrog_v_step for all particles
		for pi in range(num_p):
			# Get the position and velocity of our particle
			pos = part_pos[pi]
			vel = part_vel[pi]

			# Set the mass of this particle to 0
			save_mass = part_mass[pi]
			part_mass[pi] = 0
			a1 = calc_neg_grad_nb(part_pos,part_mass,pos)

			# Do a step of leapfrog integration on this particle
			leapfrog_v_step(vel,dt,a0_vec[pi],a1)

			# Reset the mass
			part_mass[pi] = save_mass

	# Save the pos and vel for the final step
	save_pos_array[num_dt] += part_pos
	save_vel_array[num_dt] += part_vel


"""-----------------------------NFW Functions-----------------------------"""


@numba.njit()
def calc_neg_grad_nfw(rho_0,r_scale,pos_nfw,pos):
	""" Calculate the negative gradient of the nfw potential at a specific
		position.

		Parameters:
			rho_0 (float): the density normalization parameter for the NFW
			r_scale (float): The scale radius for the NFW profile
			pos_nfw (np.array): The 3D position of the NFW
			pos (np.array): The 3D position to calculate the gradient at

		Returns:
			(np.array): The 3D negative gradient of the potential
	"""
	norm = 4 * np.pi * G * rho_0 * r_scale**3
	r2 = np.sum(np.square(pos_nfw-pos))
	r = np.sqrt(r2)
	r_hat = (pos_nfw-pos)/r

	return norm*(1/r2 * np.log(1+r/r_scale) - 1/(r_scale+r)*1/r) * r_hat
