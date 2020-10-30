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
	1)	Change the units being used to something more natural and reflect
		that in the comments
	2)	Include variable force softening scale in the integrator codes
	3)	Write code to allow for multiple NFWs?
	4) 	Discuss unit choices and purpose in an opening statement.
"""

import numba
import numpy as np
import math

# Declare some global variables (400 km^2*300 kpc/(1e13 M_sun*s^2))
G = 0.8962419740798497


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
	"""	Calculate the negative gradient of the n-body potential at a specific
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

	# We need to store the current potential for the particles before conducting
	# the update
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
	"""	Calculate the negative gradient of the nfw potential at a specific
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


@numba.njit()
def leapfrog_int_nfw(pos_init,vel_init,rho_0,r_scale,pos_nfw_array,dt,
	save_pos_array,save_vel_array):
	"""	Integrate rotation through an NFW potential

		Parameters:
			pos_init (np.array): 3D initial position of the particle. Will
				be modified.
			vel_init (np.array): 3D initial velocity of the particle. Will
				be modified.
			rho_0 (float): The amplitude of the NFW (in unites of mass)
			r_scale (float): The scale radius of the NFW
			pos_nfw_array (np.array): The num_dt*3D position of the NFW at each
				time step. This array will be used to determine the number of
				integration steps.
			dt (float): The size of a timestep
			save_pos_array (np.array): A (num_dt+1)*N*3D array where the
				positions will be saved.
			save_vel_array (np.array): A (num_dt+1)*N*3D array where the
				velocities will be saved.
	"""
	# Clear save arrays in case they have values
	save_pos_array *= 0
	save_vel_array *= 0
	a0 = np.zeros(3,dtype=np.float64)
	a1 = np.zeros(3,dtype=np.float64)
	num_dt = len(pos_nfw_array)

	for ti in range(num_dt):
		# Save the pos and vel at this step
		save_pos_array[ti] += pos_init
		save_vel_array[ti] += vel_init

		# First get the force for the particle at the current position
		a0 += calc_neg_grad_nfw(rho_0,r_scale,pos_nfw_array[ti],pos_init)

		# Do the fist integration step.
		leapfrog_p_step(pos_init,vel_init,dt,a0)

		# Calculate the force on the particle at the new position.
		a1 += calc_neg_grad_nfw(rho_0,r_scale,pos_nfw_array[ti],pos_init)

		# Do a step of leapfrog integration on this particle
		leapfrog_v_step(vel_init,dt,a0,a1)

		# Reset the force vectors
		a0 *= 0
		a1 *= 0

	# Save the pos and vel for the final step
	save_pos_array[num_dt] += pos_init
	save_vel_array[num_dt] += vel_init


"""-----------------Dynamical Friction + NFW Functions----------------------"""


@numba.njit()
def calc_neg_grad_f_dyn(vel,rho,m_sub,m_host,sigma_v):
	"""	Calculate the negative gradient caused by dynamical friction in the
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
	a_dyn *=(math.erf(x)-2*x/np.sqrt(np.pi)*np.exp(-x**2))

	return a_dyn
