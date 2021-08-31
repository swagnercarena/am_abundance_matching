# -*- coding: utf-8 -*-
"""
Functions for integrating subhalos in the host potential.

Units: In order to avoid numerical issues in the calculation, the units have
	been chosen to work well for objects ranging in mass from 1e9-1e15 M_sun.
	Therefore the units are as follows:
	Mass - 1e13 M_sun
	Radius - 300 kpc
	Velocity - 400 km/s
	Time - By virtue of the other units, time is in units of s*300kpc/400km
		or 2.31425819e16 seconds
"""
from avocet import integrator, conversion
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import root_scalar
from colossus.halo import profile_nfw, mass_so


class TrajectoriesIntegrator():
	"""
	Class for integrating subhalos within a host halo potential.

	Args:
		trj_host (trj.Trajectories): A trajectories object that has a get_halo
			and get_host property that will be used to extract a numpy array
			object with the rockstar output for the halos.
		cosmo (colossus.cosmology.cosmology.Cosmology): An instance of
			the colussus cosmology object that will be used for cosmology
			calculations.
		use_um (bool): If true use the universe machine choices for integration
	"""

	# Array of C_VALS to try for concentration calculation
	G_TRADITIONAL = 4.30091e-6
	XMAX = 2.16258
	C_VALS = np.linspace(1,50,int(2e2))

	def __init__(self,trj_host,cosmo,use_um=False):

		# Store the boolean on universe machine
		self.use_um = use_um

		# Store the trajectory and cosmology objects
		self.trj_host = trj_host
		self.cosmo = cosmo

		# Grab the host
		self.host = trj_host.get_host()

		# Grab the snapshots where the host is identified
		self.host_snaps = self.host['snap'][self.host['mvir']>0]

		# Get the the host redshifts and corresponding times.
		self.host_scales = trj_host.scale[self.host_snaps]
		zs = 1/self.host_scales-1
		# Convert from Gyr to 2.31425819e16 seconds. See note at top of file.
		self.host_times = conversion.convert_unit_gigayear(
			cosmo.lookbackTime(zs))

		# Extract all of the host information for all of the snapshots.
		self.host_pos = self.host['x'][self.host_snaps]  # comoving Mpc/h
		self.host_mass = self.host['mvir'][self.host_snaps]  # Msun/h
		self.host_vmax = self.host['vmax'][self.host_snaps]  # km/s
		self.host_vel = self.host['v'][self.host_snaps]  # km/s
		self.host_r_s_rockstar = (self.host['rs'][self.host_snaps]/1e3*
			self.host_scales)  # Mpc / h

		# Now calculate the NFW amplitude, scale factor, and concentration for
		# the host.
		self.host_rho = np.zeros(self.host_mass.shape)  # M_sun / kpc^3 * h^2
		self.host_r_s = np.zeros(self.host_mass.shape)  # Mpc / h
		self.host_c = np.zeros(self.host_mass.shape)
		self.host_r_vir = np.zeros(self.host_mass.shape)  # Mpc / h

		# Populate the array of NFW parameters.
		for i in range(len(self.host_mass)):
			if use_um:
				self.host_r_s[i] = self.host_r_s_rockstar[i]
				c, rho = self._convert_m_r_s(self.host_mass[i],
					self.host_r_s[i],zs[i])
				self.host_c[i] = c
				self.host_rho[i] = rho
			else:
				c, rho, r_s, v_vir, r_vir = self._convert_m_vmax(
					self.host_mass[i],self.host_vmax[i],zs[i])
				self.host_c[i] = c
				self.host_rho[i] = rho
				self.host_r_s[i] = r_s

		# Calculate the virial radius
		self.host_r_vir = self.host_r_s*self.host_c  # Mpc/h

		# Convert to the units of our integration code.
		self.host_rho_int = conversion.convert_units_density(
			self.host_rho*cosmo.h**2)
		self.host_r_s_int = conversion.convert_unit_Mpc(
			self.host_r_s/cosmo.h)
		self.host_pos_int = conversion.convert_unit_Mpc(
			(self.host_pos.T / cosmo.h *self.host_scales).T)

	def _convert_m_r_s(self,mass,r_s,z):
		"""	Convert from mass and r_s to profile amplitude and concentration
		for an NFW

		Args:
			mass (float): Mass of the NFW in units of M_sun/h
			r_s (float): scale radius of the NFW in units of Mpc/h
			z (float): The redshift of the NFW

		Returns,
			((float,float,float,float)): A tuple containing the concentration
				of the NFW, and the amplitude of the NFW in units of
				M_sun*h^2/kpc^3.
		"""
		# First calculate the concentration
		rho_overd = mass_so.densityThreshold(z,'vir')
		c = (mass/(4*np.pi/3*rho_overd))**(1/3) * 1/(r_s*1e3)  # r_s in kpc

		# Use colossus for the rho parameter (a little sloppy but quick)
		rho, _ = profile_nfw.NFWProfile.fundamentalParameters(M=mass,c=c,z=z,
			mdef='vir')

		return c, rho

	def _convert_m_vmax(self,mass,v_max,z):
		"""	Convert from mass and v_max to profile amplitude and concentration
		for an NFW

		Args:
			mass (float): Mass of the NFW in units of M_sun/h
			v_max (float): v_max of the NFW in units of km/s
			z (float): The redshift of the NFW

		Returns,
			((float,float,float,float)): A tuple containing the concentration
				of the NFW, the amplitude of the NFW in units of
				M_sun*h^2/kpc^3, and the scale radius in units of Mpc/h.
		"""

		# Create our f function
		def f(x):
			return np.log(1+x)-x/(1+x)

		rho_overd = mass_so.densityThreshold(z,'vir')

		# Create our v_max function
		def v_max_dif(c,mass,v_max):
			if c < 1:
				return 100
			r_vir  = (mass/(4*np.pi/3*rho_overd))**(1/3)  # In kpc/h
			v_max_calc = np.sqrt(self.__class__.G_TRADITIONAL * mass / r_vir * f(
				self.__class__.XMAX)/f(c)*c/self.__class__.XMAX)
			return v_max_calc-v_max

		c = root_scalar(v_max_dif,args=(mass,v_max),x0=6,x1=10).root

		# Use colossus to convert to the amplitude and scale radius.
		rho, r_s = profile_nfw.NFWProfile.fundamentalParameters(M=mass,c=c,z=z,
			mdef='vir')
		v_vir = profile_nfw.NFWProfile(M=mass,c=c,z=z,
			mdef='vir').circularVelocity(c*r_s)

		# Divide the scale radius by 1000 to convert it to Mpc/h from kpc/h
		r_s /= 1000
		r_vir = r_s * c

		return c, rho, r_s, v_vir, r_vir

	def load_subhalo(self,sub_id,num_t_step,dt=None):
		""" Load the subhalo properties for the given sub_id

		Args:
			sub_id (int): The id of the subhalo to integrate forward in the
				potential.
			num_t_step (int): The number of integration steps to use between
				redshift of accretion and redshift 0.
			dt (float): The timestep to use in the integration. This will
				overwrite num_t_step.
		"""

		# Get the subhalo we want to integrate.
		self.sub_halo = self.trj_host.get_halo(sub_id)

		# Subhalos go from smallest to largest snapshot so reverse the order to
		# agree with other conventions.
		self.sub_snaps = self.sub_halo['snap'][::-1]

		# For simplicity, ignore subhalos that appear before the host or do not
		# survive until redshift 0.
		if (np.max(self.sub_snaps) < np.max(self.host_snaps) or
			np.min(self.sub_snaps) < np.min(self.host_snaps)):
			return None,None

		# Pull out the subhalo information from the trajectory file.
		self.sub_scales = self.trj_host.scale[self.sub_snaps]
		self.sub_pos = self.sub_halo['x'][::-1]  # comoving Mpc/h
		self.sub_dx = self.sub_halo['dx'][::-1]  # comoving Mpc/h
		# Physical Mpc/h
		self.sub_dx_physical = (self.sub_scales * self.sub_dx.T).T
		self.sub_vel = self.sub_halo['v'][::-1]  # km/s

		# Get the the indices of the host to consider
		self.host_sub_ind = self.host['snap'] >= np.min(self.sub_halo['snap'])
		self.host_sub_ind = self.host_sub_ind[self.host_snaps]

		# Pick the snapshot to start integration the first time the subhalo
		# enters the virial radius
		self.int_start = np.where(np.sqrt(np.sum(np.square(
			self.sub_dx_physical),axis=-1))<self.host_r_vir[
			self.host_sub_ind])[0][0]

		# Convert the subhalo position and velocity as well
		self.sub_pos_int = conversion.convert_unit_Mpc((self.sub_pos.T /
			self.cosmo.h * self.sub_scales).T)
		self.sub_vel_int = conversion.convert_units_kms(self.sub_vel)

		# Set up the array of integration times
		self.sub_times = self.host_times[self.host_sub_ind][self.int_start:]
		self.time_end = self.host_times[self.host_sub_ind][self.int_start]
		self.time_start = np.min(self.host_times)
		# If dt is specified use it to define the number of integartion
		# steps.
		if dt is not None:
			num_t_step = int((self.time_end - self.time_start)//dt)
		self._num_t_step = num_t_step
		# Even if dt is specified, getting a integer number of integration
		# steps will not allow for that exact dt.
		self.dt = (self.time_end - self.time_start)/self._num_t_step
		self.times_int = np.linspace(self.time_start,self.time_end,
			self._num_t_step)[::-1]
		self.scales_int = 1/(1+self.cosmo.lookbackTime(
			self.times_int/conversion.convert_unit_gigayear(1),
			inverse=True))

		# Interpolate the subhalo properties for integation
		self.pos_nfw_array = np.zeros((self._num_t_step,3))
		for i in range(3):
			self.pos_nfw_array[:,i] = interp1d(self.host_times,
				self.host_pos_int[:,i],kind='cubic')(self.times_int)
		self.rho_0_array = interp1d(self.host_times,self.host_rho_int)(
			self.times_int)
		self.r_scale_array = interp1d(self.host_times,self.host_r_s_int)(
			self.times_int)

		# Get the box length and convert it to physical coordinates
		self.box_length_array = conversion.convert_unit_Mpc(self.trj_host.L/
			self.cosmo.h * self.scales_int)

	def integrate_sub_hf(self,sub_id,num_t_step=int(1e4),dt=None):
		""" Integrate the path of a subhalo assuming an NFW potential for the
		main host and accounting for the hubble flow.

		Args:
			sub_id (int): The id of the subhalo to integrate forward in the
				potential.
			num_t_step (int): The number of integration steps to use between
				redshift of accretion and redshift 0.
			dt (float): The timestep to use in the integration. This will
				overwrite num_t_step.

		Returns:
			((np.array,np.array)): A tuple containing two numpy arrays, the
				first with the position vector at each time step and the
				second with the velocity vector at each time step.

		Notes:
			Integration begins at the first snapshot where the subhalo is
			within the virial radius of the host.
		"""

		self.load_subhalo(sub_id,num_t_step,dt)

		# For simplicity, ignore subhalos that appear before the host or do not
		# survive until redshift 0.
		if (np.max(self.sub_snaps) < np.max(self.host_snaps) or
			np.min(self.sub_snaps) < np.min(self.host_snaps) or
			len(self.sub_snaps)<2):
			return None,None

		# Set up that arrays that will contain the integration outputs
		save_pos_array = np.zeros((self._num_t_step+1,3))
		save_vel_array = np.zeros((self._num_t_step+1,3))

		# Set up the array of hubble flow values at each integration time
		hf_int =  conversion.convert_units_kms(self.cosmo.Hz(
			1/self.scales_int-1))/conversion.convert_unit_Mpc(1)

		# Integrate the subhalo. The save_pos_array/save_vel_array are modified
		# in place.
		integrator.leapfrog_int_nfw_hf(self.sub_pos_int[self.int_start],
			self.sub_vel_int[self.int_start],self.rho_0_array,
			self.r_scale_array,self.pos_nfw_array,hf_int,self.dt,save_pos_array,
			save_vel_array,box_length_array=self.box_length_array)

		# Transform back into comoving Mpc/h units and km/s.
		save_pos_comv = (save_pos_array[:-1].T * self.cosmo.h /
			self.scales_int*0.3).T
		save_vel_comv = save_vel_array[:-1]*400

		return save_pos_comv, save_vel_comv

	def integrate_sub_rf(self,sub_id,num_t_step=int(1e4),dt=None):
		""" Integrate the path of a subhalo assuming an NFW potential for the
		main host and integrating in the host's frame of reference.

		Args:
			sub_id (int): The id of the subhalo to integrate forward in the
				potential.
			num_t_step (int): The number of integration steps to use between
				redshift of accretion and redshift 0.
			dt (float): The timestep to use in the integration. This will
				overwrite num_t_step.

		Returns:
			((np.array,np.array)): A tuple containing two numpy arrays, the
				first with the position vector at each time step and the
				second with the velocity vector at each time step.

		Notes:
			Integration begins at the first snapshot where the subhalo is
			within the virial radius of the host.
		"""

		self.load_subhalo(sub_id,num_t_step,dt)

		# For simplicity, ignore subhalos that appear before the host or do not
		# survive until redshift 0.
		if (np.max(self.sub_snaps) < np.max(self.host_snaps) or
			np.min(self.sub_snaps) < np.min(self.host_snaps) or
			len(self.sub_snaps)<2):
			return None,None

		# Set up that arrays that will contain the integration outputs
		save_pos_array = np.zeros((self._num_t_step+1,3))
		save_vel_array = np.zeros((self._num_t_step+1,3))

		# We need slightly different integration variables
		self.sub_pos_int = conversion.convert_unit_Mpc(((self.sub_pos.T -
			self.host_pos[self.host_sub_ind].T)/ self.cosmo.h *
			self.sub_scales).T)
		self.sub_vel_int = conversion.convert_units_kms(self.sub_vel -
			self.host_vel[self.host_sub_ind])

		# Interpolate the subhalo properties for integation
		self.pos_nfw_array = np.zeros((self._num_t_step,3))  # Keep this at 0.

		# Integrate the subhalo. The save_pos_array/save_vel_array are modified
		# in place.
		integrator.leapfrog_int_nfw(self.sub_pos_int[self.int_start],
			self.sub_vel_int[self.int_start],self.rho_0_array,
			self.r_scale_array,self.pos_nfw_array,self.dt,save_pos_array,
			save_vel_array,box_length_array=None)

		# Transform back into comoving Mpc/h units
		save_pos_comv = (save_pos_array[:-1].T * self.cosmo.h /
			self.scales_int*0.3).T
		for i in range(3):
			save_pos_comv[:,i] += interp1d(self.host_times,
				self.host_pos[:,i],kind='cubic')(self.times_int)
		save_vel_comv = save_vel_array[:-1]*400
		for i in range(3):
			save_vel_comv[:,i] += interp1d(self.host_times,
				self.host_vel[:,i],kind='cubic')(self.times_int)

		return save_pos_comv, save_vel_comv
