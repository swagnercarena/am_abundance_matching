# -*- coding: utf-8 -*-
"""
Functions for integrating a subhalo within a host potential

Units: In order to avoid numerical issues in the calculation, the units have
	been chosen to work well for objects ranging in mass from 1e9-1e15 M_sun.
	Therefore the units are as follows:
	Mass - 1e13 M_sun
	Radius - 300 kpc
	Velocity - 400 km/s
	Time - By virtue of the other units, time is in units of s*300kpc/400km
		or 2.31425819e16 seconds

TODO:
	2)	Include variable force softening scale in the integrator codes
	3)	Write code to allow for multiple NFWs?
	4) 	Discuss unit choices and purpose in an opening statement.
"""

import math
import numba
import numpy as np

# Declare some global variables (400 km^2*300 kpc/(1e13 M_sun*s^2))
G = 0.8962419740798497


@numba.njit()
def leapfrog_p_step(x0,v0,dt):
	"""
	Given the current position and velocity of a particle conduct one drift
	step of leapfrog integration.

	Parameters:
		x0 (np.array): The current 3d position. Modified in place to x1
		v0 (np.array): The current 3d velocity.
		dt (float): The timestep for integration
	"""
	x0 += v0*dt


@numba.njit()
def leapfrog_p_step_cosmo(x0,v0,dt,scale):
	"""
	Given the current position and velocity of a particle conduct one drift
	step of leapfrog integration.

	Parameters:
		x0 (np.array): The current 3d position. Modified in place to x1
		v0 (np.array): The current 3d velocity.
		dt (float): The timestep for integration

	Notes:
		Position inputs are assumed to be in comoving coordinates. This code
		assumes the change in the scale factor over a time step is negligible.
	"""
	x0 += v0*dt/scale**2


@numba.njit()
def leapfrog_v_step(v0,dt,a0):
	"""
	Given the current velocity and acceleration of a particle, conduct one
	kick step of leapfrog integration.

	Parameters:
		v0 (np.array): The current 3d velocity. Modified in place to v1
		dt (float): The timestep for integration
		a0 (np.array): The 3d gradient of the potential.
	"""
	v0 += a0*dt


@numba.njit()
def leapfrog_v_step_cosmo(v0,dt,a0,scale):
	"""
	Given the current velocity and acceleration of a particle, conduct one
	kick step of leapfrog integration.

	Parameters:
		v0 (np.array): The current 3d velocity. Modified in place to v1
		dt (float): The timestep for integration
		a0 (np.array): The 3d gradient of the potential.
		scale (float): The scale factor

	Notes:
		This code assumes the change in the scale factor over a time
		step is negligible.
	"""
	v0 += a0*dt/scale


"""----------------------------N Body Functions---------------------------"""


@numba.njit()
def calc_neg_grad_nb(part_pos,part_mass,pos):
	"""
	Calculate the negative gradient of the n-body potential at a specific
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
	"""
	Integrate the n body problem using kick-drift-kick leapfrog

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

	# We need to store the potential for the particles so we will allocate
	# the variable
	ai = np.zeros((3),dtype=np.float64)

	# Variables we'll use for inplace calculation
	save_mass = 0

	# Clear save arrays in case they have values
	save_pos_array *= 0
	save_vel_array *= 0

	for ti in range(num_dt):
		# Save the pos and vel at this step
		save_pos_array[ti] += part_pos
		save_vel_array[ti] += part_vel

		# Kick step
		for pi in range(num_p):
			# Get the position and velocity of our particle
			pos = part_pos[pi]
			vel = part_vel[pi]

			# Set the mass of this particle to 0
			save_mass = part_mass[pi]
			part_mass[pi] = 0
			ai = calc_neg_grad_nb(part_pos,part_mass,pos)

			# Do a step of leapfrog integration on this particle
			leapfrog_v_step(vel,dt/2,ai)

			# Reset the mass
			part_mass[pi] = save_mass

		# Drift step
		for pi in range(num_p):
			# Update the position of our particles
			pos = part_pos[pi]
			vel = part_vel[pi]

			leapfrog_p_step(pos,vel,dt)

		# Kick step
		for pi in range(num_p):
			# Get the position and velocity of our particle
			pos = part_pos[pi]
			vel = part_vel[pi]

			# Set the mass of this particle to 0
			save_mass = part_mass[pi]
			part_mass[pi] = 0
			ai = calc_neg_grad_nb(part_pos,part_mass,pos)

			# Do a step of leapfrog integration on this particle
			leapfrog_v_step(vel,dt/2,ai)

			# Reset the mass
			part_mass[pi] = save_mass

	# Save the pos and vel for the final step
	save_pos_array[num_dt] += part_pos
	save_vel_array[num_dt] += part_vel


"""-----------------------------NFW Functions-----------------------------"""


@numba.njit()
def calc_neg_grad_nfw(rho_0,r_scale,pos_nfw,pos,epsilon=1e-5):
	"""
	Calculate the negative gradient of the nfw potential at a specific
	position.

	Parameters:
		rho_0 (float): the density normalization parameter for the NFW
		r_scale (float): The scale radius for the NFW profile
		pos_nfw (np.array): The 3D position of the NFW
		pos (np.array): The 3D position to calculate the gradient at
		epsilon (float): A force softening scale in units of (300kpc)^2.

	Returns:
		(np.array): The 3D negative gradient of the potential
	"""
	norm = 4 * np.pi * G * rho_0 * r_scale**3
	# Add epsilon to avoid collisions
	r2 = np.sum(np.square(pos_nfw-pos))+epsilon
	r = np.sqrt(r2)
	# Don't include epsilon in r_hat calculation
	r_hat = (pos_nfw-pos)/np.sqrt(r2-epsilon)

	return norm*(1/r2 * np.log(1+r/r_scale) - 1/(r_scale+r)*1/r) * r_hat


@numba.njit()
def leapfrog_int_nfw(pos_init,vel_init,rho_0_array,r_scale_array,pos_nfw_array,
	dt,save_pos_array,save_vel_array):
	"""
	Integrate rotation through an NFW potential using kick-drift-kick leapfrog

	Parameters:
		pos_init (np.array): 3D initial position of the particle. Will
			be modified.
		vel_init (np.array): 3D initial velocity of the particle. Will
			be modified.
		rho_0_array (np.array): A num_dt array with the amplitude of the
			NFW (in unites of mass) at each time step.
		r_scale_array (np.array): A num_dt array with the the scale radius
			of the NFW at each time step.
		pos_nfw_array (np.array): The num_dt*3D position of the NFW at each
			time step. This array will be used to determine the number of
			integration steps.
		dt (float): The size of a timestep
		save_pos_array (np.array): A (num_dt+1)*N*3D array where the
			positions will be saved.
		save_vel_array (np.array): A (2*num_dt+1)*N*3D array where the
			velocities will be saved.
	"""
	# Clear save arrays in case they have values
	save_pos_array *= 0
	save_vel_array *= 0

	# Allocate array to store acceleration
	ai = np.zeros(3,dtype=np.float64)

	num_dt = len(pos_nfw_array)

	for ti in range(num_dt):
		# Save the pos and vel at this step
		save_pos_array[ti] += pos_init
		save_vel_array[ti] += vel_init

		# Kick step
		ai += calc_neg_grad_nfw(rho_0_array[ti],r_scale_array[ti],
			pos_nfw_array[ti],pos_init)
		leapfrog_v_step(vel_init,dt/2,ai)

		# Reset the force vectors
		ai *= 0

		# Drift step
		leapfrog_p_step(pos_init,vel_init,dt)

		# Kick step.
		ai += calc_neg_grad_nfw(rho_0_array[ti],r_scale_array[ti],
			pos_nfw_array[ti],pos_init)
		leapfrog_v_step(vel_init,dt/2,ai)

		# Reset the force vectors
		ai *= 0

	# Save the pos and vel for the final step
	save_pos_array[num_dt] += pos_init
	save_vel_array[num_dt] += vel_init


@numba.njit()
def leapfrog_int_nfw_hf(pos_init,vel_init,rho_0_array,r_scale_array,
	pos_nfw_array,hf,dt,save_pos_array,save_vel_array):
	"""
	Integrate rotation through an NFW potential using kick-drift-kick leapfrog

	Parameters:
		pos_init (np.array): 3D initial position of the particle. Will
			be modified.
		vel_init (np.array): 3D initial velocity of the particle. Will
			be modified.
		rho_0_array (np.array): A num_dt array with the amplitude of the
			NFW (in unites of mass) at each time step.
		r_scale_array (np.array): A num_dt array with the the scale radius
			of the NFW at each time step.
		pos_nfw_array (np.array): The num_dt*3D position of the NFW at each
			time step. This array will be used to determine the number of
			integration steps.
		hf (np.array): The num_dt*3D array of the hubble flow velocity at
			each time step
		dt (float): The size of a timestep
		save_pos_array (np.array): A (num_dt+1)*N*3D array where the
			positions will be saved.
		save_vel_array (np.array): A (2*num_dt+1)*N*3D array where the
			velocities will be saved.
	"""
	# Clear save arrays in case they have values
	save_pos_array *= 0
	save_vel_array *= 0

	# Allocate array to store acceleration
	ai = np.zeros(3,dtype=np.float64)

	num_dt = len(pos_nfw_array)

	for ti in range(num_dt):
		# Save the pos and vel at this step
		save_pos_array[ti] += pos_init
		save_vel_array[ti] += vel_init

		# Kick step
		ai += calc_neg_grad_nfw(rho_0_array[ti],r_scale_array[ti],
			pos_nfw_array[ti],pos_init)
		leapfrog_v_step(vel_init,dt/2,ai)

		# Reset the force vectors
		ai *= 0

		# Drift step
		hf_vel = hf[ti]*pos_init
		leapfrog_p_step(pos_init,vel_init+hf_vel,dt)

		# Kick step.
		ai += calc_neg_grad_nfw(rho_0_array[ti],r_scale_array[ti],
			pos_nfw_array[ti],pos_init)
		leapfrog_v_step(vel_init,dt/2,ai)

		# Reset the force vectors
		ai *= 0

	# Save the pos and vel for the final step
	save_pos_array[num_dt] += pos_init
	save_vel_array[num_dt] += vel_init


@numba.njit()
def leapfrog_int_nfw_cosmo(pos_init,vel_init,rho_0_array,r_scale_array,
	scale_factor_array,pos_nfw_array,dt,save_pos_array,save_vel_array):
	"""
	Integrate rotation through an NFW potential using kick-drift-kick
	leapfrog in an expanding universe.

	Parameters:
		pos_init (np.array): 3D initial position of the particle. Will
			be modified. Must be in comoving units.
		vel_init (np.array): 3D initial velocity of the particle. Will
			be modified.
		rho_0_array (np.array): A num_dt array with the amplitude of the
			NFW (in unites of mass) at each time step.
		r_scale_array (np.array): A num_dt array with the the scale radius
			of the NFW at each time step.
		scale_factor_array (np.array): A num_dt array with the scale factor
			of the universe at each time step.
		pos_nfw_array (np.array): The num_dt*3D position of the NFW at each
			time step. This array will be used to determine the number of
			integration steps. Must be in comoving units.
		dt (float): The size of a timestep
		save_pos_array (np.array): A (num_dt+1)*N*3D array where the
			positions will be saved. Will be in comoving units.
		save_vel_array (np.array): A (num_dt+1)*N*3D array where the
			velocities will be saved.

	Notes:
		Position inputs must be in comoving coordinates.
	"""
	# Clear save arrays in case they have values
	save_pos_array *= 0
	save_vel_array *= 0

	# Allocate array to store acceleration
	ai = np.zeros(3,dtype=np.float64)

	num_dt = len(pos_nfw_array)

	for ti in range(num_dt):
		# Save the pos and vel at this step
		save_pos_array[ti] += pos_init
		save_vel_array[ti] += vel_init

		scale = scale_factor_array[ti]

		# Kick step. Convert to physical to calculate gradients.
		ai += calc_neg_grad_nfw(rho_0_array[ti],r_scale_array[ti],
			pos_nfw_array[ti]*scale,pos_init*scale)*scale**2
		leapfrog_v_step_cosmo(vel_init,dt/2,ai,scale)

		# Reset the force vectors
		ai *= 0

		# Drift step
		leapfrog_p_step_cosmo(pos_init,vel_init,dt,scale)

		# Kick step. Convert to physical to calculate gradients.
		ai += calc_neg_grad_nfw(rho_0_array[ti],r_scale_array[ti],
			pos_nfw_array[ti]*scale,pos_init*scale)*scale**2
		leapfrog_v_step_cosmo(vel_init,dt/2,ai,scale)

		# Reset the force vectors
		ai *= 0

	# Save the pos and vel for the final step
	save_pos_array[num_dt] += pos_init
	save_vel_array[num_dt] += vel_init


"""-----------------Dynamical Friction + NFW Functions----------------------"""


@numba.njit()
def calc_neg_grad_f_dyn(vel,rho,m_sub,m_host,sigma_v):
	"""
	Calculate the negative gradient caused by dynamical friction in the
	potential.

	Parameters:
		vel (np.array): The 3D velocity of the substructure in question
		rho (float): The matter density at the position of the
			substructure
		m_sub (float): The mass of the substructure
		m_host (float): The mass of the host halo
		sigma_v (float): The velocity dispersion of the halo at the
			position of the substructure.

	Returns:
		(np.array): The 3D negative gradient of the dynamical
			friction 'potential'.
	"""
	# Calculate some velocity qunatities we will use.
	v_2 = np.sum(np.square(vel))
	v_mag = np.sqrt(v_2)

	# First calculate the Coulomb logarithm
	ln_lambda = np.log(m_host) - np.log(m_sub)

	# Calculate the dynamical friction variable X
	x = v_mag / (np.sqrt(2)*sigma_v)

	# Put everything together into our dynamical friction term
	a_dyn = vel/(v_2*v_mag)
	a_dyn *= -4*np.pi*G**2*m_sub*rho*ln_lambda
	a_dyn *= (math.erf(x)-2*x/np.sqrt(np.pi)*np.exp(-x**2))

	return a_dyn


@numba.njit()
def calc_mass_loss(vel,m_sub,m_host,rho_vir,pos_nfw,pos):
	"""
	Calculate the mass loss caused by stripping from the host halo.

	Parameters:
		vel (np.array): The 3D velocity of the substructure in question
		m_sub (float): The mass of the substructure
		m_host (float): The mass of the host halo
		rho_vir (float): The density used to define the virial
			density.
		pos_nfw (np.array): The 3D position of the NFW
		pos (np.array): The 3D position to calculate the gradient at

	Returns:
		(float): The derivative of the mass loss as a function of time.
	"""
	# First check if the subhalo is infalling
	r = pos_nfw-pos
	# If outoing no mass is lost
	if np.dot(r,vel)<0:
		return 0
	# If infalling mass is being lost.
	else:
		# Calculate the dynamical time.
		one_over_t_dyn = np.sqrt(4*np.pi*G*rho_vir/3)

		# Now calculate the mass derivative
		return -1.18*m_sub*one_over_t_dyn*(m_sub/m_host)**(0.07)


@numba.njit()
def nfw_sigma_v(r_scale,v_max,pos_nfw,pos):
	"""
	Get the velocity dispersion for an NFW at a given position based on
	https://iopscience.iop.org/article/10.1086/378797/pdf.

	Parameters:
		r_scale (float): The scale radius for the NFW profile
		v_max (np.array): The magnitude of the maximum velocity for the
			host halo
		pos_nfw (np.array): The 3D position of the NFW
		pos (np.array): The 3D position to calculate the gradient at

	Returns:
		(float): The derivative of the mass loss as a function of time.
	"""
	# The approximate function is written as a ratio of the radius to the
	# scale radius.
	r = np.sqrt(np.sum(np.square(pos_nfw-pos)))
	x = r/r_scale

	# The approximate function complete with magic numbers
	return v_max * (1.4393*x**(0.354)) / (1+1.1756*x**(0.725))


@numba.njit()
def nfw_rho(r_scale,rho_0,pos_nfw,pos):
	"""
	Calculate the density of the NFW at a given position.

	Parameters:
		r_scale (float): The scale radius for the NFW profile
		v_max (np.array): The magnitude of the maximum velocity for the
			host halo
		pos_nfw (np.array): The 3D position of the NFW
		pos (np.array): The 3D position to calculate the gradient at

	Returns:
		(float): The density at position pos.
	"""
	# Get the radial distance in the profile
	r = np.sqrt(np.sum(np.square(pos_nfw-pos)))
	return rho_0/(r/r_scale*(1+r/r_scale)**2)


@numba.njit()
def leapfrog_int_nfw_f_dyn_hf(pos_init,vel_init,m_sub,rho_0_array,
	r_scale_array,pos_nfw_array,hf,m_nfw_array,v_max_nfw_array,rho_vir,dt,
	save_pos_array,save_vel_array):
	"""
	Integrate rotation through an NFW potential with dynamical friction

	Parameters:
		pos_init (np.array): 3D initial position of the particle. Will
			be modified.
		vel_init (np.array): 3D initial velocity of the particle. Will
			be modified.
		m_sub (float): The mass of the incoming substructure
		rho_0_array (np.array): A num_dt array with the amplitude of the
			NFW (in unites of mass) at each time step.
		r_scale_array (np.array): A num_dt array with the yhe scale radius
			of the NFW at each time step.
		pos_nfw_array (np.array): The num_dt*3D position of the NFW at each
			time step. This array will be used to determine the number of
			integration steps.
		hf_int (np.array): The num_dt*3D array of the hubble flow velocity at
			each time step
		m_nfw_array (np.array): A num_dt array with the mass of the NFW at
			each time step.
		v_max_nfw_array (np.array): A num_dt array with the maximum
			velocity of the NFW at each timestep.
		rho_vir (float): The density used to define the virial radius /
			virial mass for the NFW.
		dt (float): The size of a timestep
		save_pos_array (np.array): A (num_dt+1)*N*3D array where the
			positions will be saved.
		save_vel_array (np.array): A (num_dt+1)*N*3D array where the
			velocities will be saved.
	"""
	# Clear save arrays in case they have values
	save_pos_array *= 0
	save_vel_array *= 0

	# Allocate array to store acceleration
	ai = np.zeros(3,dtype=np.float64)

	num_dt = len(pos_nfw_array)

	for ti in range(num_dt):
		# Save the pos and vel at this step
		save_pos_array[ti] += pos_init
		save_vel_array[ti] += vel_init

		# First get the force for the particle at the current position
		ai += calc_neg_grad_nfw(rho_0_array[ti],r_scale_array[ti],
			pos_nfw_array[ti],pos_init)

		# Calculate the terms required for the dynamical friction term.
		sigma_v = nfw_sigma_v(r_scale_array[ti],v_max_nfw_array[ti],
			pos_nfw_array[ti],pos_init)
		rho = nfw_rho(rho_0_array[ti],r_scale_array[ti],pos_nfw_array[ti],
			pos_init)
		ai += calc_neg_grad_f_dyn(vel_init,rho,m_sub,m_nfw_array[ti],sigma_v)

		# Kick step
		leapfrog_v_step(vel_init,dt/2,ai)

		# Reset the force vector
		ai *= 0

		# Also calculate the mass loss
		m_sub += calc_mass_loss(vel_init,m_sub,m_nfw_array[ti],
			rho_vir,pos_nfw_array[ti],pos_init)*dt
		# Make sure the mass loss has not sent the mass to 0. A lower limit
		# of the mass of the sun for a subhalo is probably more than
		# generous enough.
		m_sub = max(m_sub,1e-13)

		# Drift step
		hf_vel = hf[ti]*pos_init
		leapfrog_p_step(pos_init,vel_init+hf_vel,dt)

		# Calculate the force on the particle at the new position.
		ai += calc_neg_grad_nfw(rho_0_array[ti],r_scale_array[ti],
			pos_nfw_array[ti],pos_init)
		sigma_v = nfw_sigma_v(r_scale_array[ti],v_max_nfw_array[ti],
			pos_nfw_array[ti],pos_init)
		rho = nfw_rho(rho_0_array[ti],r_scale_array[ti],pos_nfw_array[ti],
			pos_init)
		ai += calc_neg_grad_f_dyn(vel_init,rho,m_sub,m_nfw_array[ti],sigma_v)
		leapfrog_v_step(vel_init,dt/2,ai)

		# Reset the force vector
		ai *= 0

	# Save the pos and vel for the final step
	save_pos_array[num_dt] += pos_init
	save_vel_array[num_dt] += vel_init
