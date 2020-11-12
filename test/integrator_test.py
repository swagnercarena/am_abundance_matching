import unittest
from avocet import integrator
import numpy as np
from galpy.orbit import Orbit
from galpy.potential import NFWPotential
import math

# Some useful conversion factors we'll use
M_sun2kg = 1.9891e30*1e13
km2m = 400*1e3
kpc2m = 300*3.086e19


class IntegratorTestsNB(unittest.TestCase):
	# Test class for n-body methods.

	def test_leapfrog_p_step(self):
		# Test that the leapfrog p_step does the desired updates.
		x0 = np.random.randn(20,3)
		v0 = np.random.randn(20,3)
		a0 = np.random.randn(20,3)
		dt = 0.2

		for i in range(len(x0)):
			x_temp = np.copy(x0[i])
			integrator.leapfrog_p_step(x0[i],v0[i],dt,a0[i])
			np.testing.assert_almost_equal(x_temp+v0[i]*dt+0.5*a0[i]*dt**2,
				x0[i])

	def test_leapfrog_v_step(self):
		# Test that the leapfrog p_step does the desired updates.
		x0 = np.random.randn(20,3)
		v0 = np.random.randn(20,3)
		a0 = np.random.randn(20,3)
		a1 = np.random.randn(20,3)
		dt = 0.2

		for i in range(len(x0)):
			v_temp = np.copy(v0[i])
			integrator.leapfrog_v_step(v0[i],dt,a0[i],a1[i])
			np.testing.assert_almost_equal(v_temp+0.5*(a0[i]+a1[i])*dt,
				v0[i])

	def test_calc_neg_grad_nb(self):
		# Test that the nbody potential code returns the expected results
		G = 6.67430e-11

		pos = np.array([1,0,0],dtype=np.float64)
		part_pos = np.array([[0,2,0],[0,2,0],[0,2,0]],dtype=np.float64)
		part_mass = np.array([1,1,1],dtype=np.float64)

		part_pos = np.array([[0,2,0]],dtype=np.float64)
		part_mass = np.array([3],dtype=np.float64)

		g1 = integrator.calc_neg_grad_nb(part_pos,part_mass,pos)
		g2 = integrator.calc_neg_grad_nb(part_pos,part_mass,pos)
		np.testing.assert_almost_equal(g1,g2)

		# Check a more complicated potential
		part_pos = np.array([[0,1,0],[0,0,2]])
		part_mass = np.array([3,4],dtype=np.float64)

		g_comp = - G*part_mass[0]*M_sun2kg/2**1.5*np.array([1,-1,0])
		g_comp += - G*part_mass[1]*M_sun2kg/5**1.5*np.array([1,0,-2])
		g_comp /= kpc2m*km2m**2
		g = integrator.calc_neg_grad_nb(part_pos,part_mass,pos)
		np.testing.assert_almost_equal(g,g_comp,decimal=6)

	def test_leapfrog_int_nb(self):
		# Try some analytically known 2 body integrals and ensure that the
		# results match expectations.
		G = 0.8962419740798497
		M = 1
		r = 1
		part_pos = np.array([[0,0,0],[0,r,0]],dtype=np.float64)
		part_vel = np.array([[0,0,0],[np.sqrt(G*M/r),0,0]],dtype=np.float64)
		part_mass = np.array([M,1e-6],dtype=np.float64)

		dt = 0.001
		period = 2*np.pi*r/np.sqrt(G*M/r)
		num_dt = int(period/dt)
		save_pos_array = np.zeros((num_dt+1,2,3))
		save_vel_array = np.zeros((num_dt+1,2,3))

		integrator.leapfrog_int_nb(part_pos,part_vel,part_mass,dt,num_dt,
			save_pos_array,save_vel_array)

		for i in range(2):
			np.testing.assert_almost_equal(save_pos_array[0,i,:],
				save_pos_array[-1,i,:],decimal=3)
			np.testing.assert_almost_equal(save_vel_array[0,i,:],
				save_vel_array[-1,i,:],decimal=3)

		# Quickly repeat the same for 5 more periods
		num_dt = int(5*period/dt)
		save_pos_array = np.zeros((num_dt+1,2,3))
		save_vel_array = np.zeros((num_dt+1,2,3))
		integrator.leapfrog_int_nb(part_pos,part_vel,part_mass,dt,num_dt,
			save_pos_array,save_vel_array)

		for i in range(2):
			np.testing.assert_almost_equal(save_pos_array[0,i,:],
				save_pos_array[-1,i,:],decimal=3)
			np.testing.assert_almost_equal(save_vel_array[0,i,:],
				save_vel_array[-1,i,:],decimal=3)

		# Now check that we get reasonable results for a e = 0.9 Kepplerian
		# orbit
		rp = 1.9
		vt = 0.005927
		period = 243
		m1 = 1e2*6.67430e-11/0.8962419740798497
		m2 = 1e7*6.67430e-11/0.8962419740798497
		rcm = m1*rp/(m1+m2)
		vcm = vt/rp*rcm

		part_pos = np.array([[rp,0,0],[0,0,0]],dtype=np.float64)
		part_vel = np.array([[0,vt-vcm,0],[0,-vcm,0]],dtype=np.float64)
		part_mass = np.array([m1,m2],dtype=np.float64)
		dt = period/2010.6
		num_dt = int(period/dt)
		save_pos = np.zeros((num_dt+1,2,3))
		save_vel = np.zeros((num_dt+1,2,3))

		integrator.leapfrog_int_nb(part_pos,part_vel,part_mass,dt,num_dt,
			save_pos,save_vel)

		for i in range(2):
			for j in range(2):
				np.testing.assert_almost_equal(save_pos_array[0,i,j],
					save_pos_array[-1,i,j],decimal=3)
				np.testing.assert_almost_equal(save_vel_array[0,i,j],
					save_vel_array[-1,i,j],decimal=3)


class IntegratorTestsNFW(unittest.TestCase):
	# Test class for NFW methods.

	def test_calc_neg_grad_nfw(self):
		# Test that calculating the nfw negative gradient returns the same
		# results as galpy (but faster hopefully :))
		r_scale = 2
		rho_0 = 0.1
		G = 0.8962419740798497
		nfw = NFWPotential(a=r_scale,amp=rho_0*G*4*np.pi*r_scale**3)

		pos_nfw = np.zeros(3,dtype=np.float64)
		pos = np.zeros(3,dtype=np.float64)

		# Start just by testing a couple of different values of radius 1,2, and
		# 3.
		thetas = np.random.rand(10).astype(np.float64)*2*np.pi
		rs = np.array([1,2,3],dtype=np.float64)
		for theta in thetas:
			for r in rs:
				# Update the position
				pos[0] = r*np.cos(theta)
				pos[1] = r*np.sin(theta)

				# Compare both magnitudes
				r_force = nfw.Rforce(r,0)
				neg_grad = integrator.calc_neg_grad_nfw(rho_0,r_scale,pos_nfw,
					pos)

				self.assertAlmostEqual(np.abs(r_force),
					np.sqrt(np.sum(np.square(neg_grad))))

				# Ensure the direction is correct
				np.testing.assert_almost_equal(
					neg_grad/np.sqrt(np.sum(np.square(neg_grad))),
					-pos/r)

	def test_leapfrog_int_nfw(self):
		# Test the circular orbits and circular orbits within a moving NFW work
		# Start with something simple like circular motion
		rho_0 = 0.1
		r_scale = 1
		G = 0.8962419740798497

		dt = 0.0001
		num_dt = int(1e5)

		pos_nfw_array = np.tile(np.zeros(3,dtype=np.float64),(num_dt,1))
		save_pos = np.zeros((num_dt+1,3))
		save_vel = np.zeros((num_dt+1,3))

		# Change the rho_0 and r_scale to a fixed array in time
		rho_0_array = rho_0*np.ones(num_dt,dtype=np.float64)
		r_scale_array = r_scale*np.ones(num_dt,dtype=np.float64)

		r_init = 20
		pos_init = np.array([r_init,0,0],dtype=np.float64)

		M_r = 4*np.pi*rho_0*r_scale**3*(np.log((r_scale+r_init)/r_scale)+
			r_scale/(r_scale+r_init) - 1)
		v_r = np.sqrt(G*M_r/r_init)
		vel_init = np.array([0,v_r,0],dtype=np.float64)
		period = 2*np.pi*np.sqrt(r_init**3/(G*M_r))
		# Get the final completed period
		last_period = int(int(num_dt*dt/period)*period//dt)

		integrator.leapfrog_int_nfw(pos_init,vel_init,rho_0_array,
			r_scale_array,pos_nfw_array,dt,save_pos,save_vel)
		# Check that it returns to the original position once per period.
		np.testing.assert_almost_equal(save_pos[0,:],save_pos[last_period,:],
			decimal=3)

		# Repeat the test with a moving NFW
		pos_nfw_array = np.tile(np.zeros(3,dtype=np.float64),(num_dt,1))
		for pi, pos in enumerate(pos_nfw_array):
			pos[0] += pi*dt

		pos_init = np.array([r_init,0,0],dtype=np.float64)
		vel_init = np.array([1,v_r,0],dtype=np.float64)
		integrator.leapfrog_int_nfw(pos_init,vel_init,rho_0_array,
			r_scale_array,pos_nfw_array,dt,save_pos,save_vel)

		np.testing.assert_almost_equal(save_pos[0,:],
			save_pos[last_period,:]-pos_nfw_array[last_period,:],
			decimal=3)

	def test_leapfrog_nfw_galpy(self):
		# Compare a more complicated set of initial conditions to the galpy
		# outputs (which are slow but known to work).

		# Set the parameters for a slightly offset circular orbit
		rho_0 = 0.1
		r_scale = 1
		G = 0.8962419740798497

		dt = 0.001
		num_dt = int(3e4)

		pos_nfw_array = np.tile(np.zeros(3,dtype=np.float64),(num_dt,1))
		save_pos = np.zeros((num_dt+1,3))
		save_vel = np.zeros((num_dt+1,3))

		# Change the rho_0 and r_scale to a fixed array in time
		rho_0_array = rho_0*np.ones(num_dt,dtype=np.float64)
		r_scale_array = r_scale*np.ones(num_dt,dtype=np.float64)

		r_init = 20
		pos_init = np.array([r_init,0,0],dtype=np.float64)

		M_r = 4*np.pi*rho_0*r_scale**3*(np.log((r_scale+r_init)/r_scale)+
			r_scale/(r_scale+r_init) - 1)
		v_r = np.sqrt(G*M_r/r_init)
		v_kick = 1e-1
		vel_init = np.array([v_kick,v_r,0],dtype=np.float64)

		integrator.leapfrog_int_nfw(pos_init,vel_init,rho_0_array,
			r_scale_array,pos_nfw_array,dt,save_pos,save_vel)

		# Compare to galpy orbits
		o=Orbit([r_init,v_kick,v_r,0,0,0])
		nfw = NFWPotential(a=r_scale,amp=rho_0*G*4*np.pi*r_scale**3)
		ts = np.linspace(0,dt*num_dt,num_dt+1)
		o.integrate(ts,nfw,method='leapfrog',dt=dt)

		np.testing.assert_almost_equal(o.x(ts),save_pos[:,0])
		np.testing.assert_almost_equal(o.y(ts),save_pos[:,1])
		np.testing.assert_almost_equal(o.z(ts),save_pos[:,2])

		# Make the kick bigger and see what happens
		v_kick = 5e-1
		pos_init = np.array([r_init,0,0],dtype=np.float64)
		vel_init = np.array([v_kick,v_r,v_kick],dtype=np.float64)
		integrator.leapfrog_int_nfw(pos_init,vel_init,rho_0_array,
			r_scale_array,pos_nfw_array,dt,save_pos,save_vel)

		# Do the same for galpy
		o=Orbit([r_init,v_kick,v_r,0,v_kick,0])
		nfw = NFWPotential(a=r_scale,amp=rho_0*G*4*np.pi*r_scale**3)
		ts = np.linspace(0,dt*num_dt,num_dt+1)
		o.integrate(ts,nfw,method='leapfrog',dt=dt)

		np.testing.assert_almost_equal(o.x(ts),save_pos[:,0])
		np.testing.assert_almost_equal(o.y(ts),save_pos[:,1])
		np.testing.assert_almost_equal(o.z(ts),save_pos[:,2])


class IntegratorTestsFricDyn(unittest.TestCase):
	# Test class for NFW methods.

	def test_calc_neg_grad_f_dyn(self):
		# Here we can't compare to galpy since their implementation of lambda
		# is different. Instead we have hand computed some correct dynamical
		# friction values and we compare to that.
		G = 0.8962419740798497
		v = np.ones(3,dtype=np.float64)
		m_sub = 1
		m_host = np.exp(4)
		rho = 1
		sigma = 1
		x = np.sqrt(3)/(np.sqrt(2)*sigma)
		x_factor = (math.erf(x)-2*x/np.sqrt(np.pi)*np.exp(-x**2))
		ln_lamba = 4

		hand_calc = -4*np.pi*G**2*m_sub/3*ln_lamba*x_factor*v/np.sqrt(3)
		np.testing.assert_almost_equal(hand_calc,
			integrator.calc_neg_grad_f_dyn(v,rho,m_sub,m_host,sigma))

	def test_calc_mass_loss(self):
		# Test the mass loss prescription work as intended.
		# First we check if the object is outoing no mass loss occurs
		pos= np.array([1,1,1],dtype=np.float64)
		pos_nfw= np.array([-1,0,2],dtype=np.float64)

		# Start with the velocity perfectly pointing out
		vel = np.array([2,1,-1],dtype=np.float64)
		G = 0.8962419740798497
		m_sub = 1e2
		m_host = 1e3
		rho_vir = 1.6

		self.assertEqual(integrator.calc_mass_loss(vel,m_sub,m_host,rho_vir,
			pos_nfw,pos),0)

		# Add some perpindicular kicks and make sure that doesn't change
		# anything
		perp1 = np.array([-1,1,-1],dtype=np.float64)
		perp2 = np.cross(perp1,vel)
		for _ in range(10):
			kick1 = np.random.randn()
			kick2 = np.random.randn()
			vel += perp1*kick1 + perp2*kick2
			self.assertEqual(integrator.calc_mass_loss(vel,m_sub,m_host,
				rho_vir,pos_nfw,pos),0)

		# Repeat the same but for ingoing
		vel = np.array([-2,-1,1],dtype=np.float64)
		t_dyn = 1/np.sqrt(4*np.pi*G*rho_vir/3)
		mass_loss = -1.18*m_sub/t_dyn
		mass_loss *= (m_sub/m_host)**(0.07)
		self.assertAlmostEqual(integrator.calc_mass_loss(vel,m_sub,m_host,
			rho_vir,pos_nfw,pos),mass_loss)

		# Random perpindicular kicks
		for _ in range(10):
			kick1 = np.random.randn()
			kick2 = np.random.randn()
			vel += perp1*kick1 + perp2*kick2
			self.assertEqual(integrator.calc_mass_loss(vel,m_sub,m_host,
				rho_vir,pos_nfw,pos),mass_loss)

	def test_nfw_sigma_v(self):
		# Test the the calculation of sigma_v returns the values we expect

		r_scale = 4
		v_max = 2
		r_values = np.linspace(0.1,r_scale,100)
		pos_nfw = np.zeros(3,dtype=np.float64)

		for r in r_values:
			x = r/r_scale
			pos = np.random.randn(3).astype(np.float64)
			pos /= np.sqrt(np.sum(np.square(pos)))
			pos *= r
			# Be extra careful about order of operations for the test
			sigma_v_hand_calc = v_max
			sigma_v_hand_calc *= 1.4393
			sigma_v_hand_calc *= x**0.354
			sigma_v_hand_calc /= (1+1.1756*(x**0.725))
			self.assertAlmostEqual(integrator.nfw_sigma_v(r_scale,
				v_max,pos_nfw,pos),sigma_v_hand_calc)

	def test_nfw_rho(self):
		# Test the the calculation of rho for an NFW returns the values we
		# expect
		r_scale = 4
		rho_0 = 20
		r_values = np.linspace(0.1,r_scale,100)
		pos_nfw = np.zeros(3,dtype=np.float64)

		for r in r_values:
			pos = np.random.randn(3).astype(np.float64)
			pos /= np.sqrt(np.sum(np.square(pos)))
			pos *= r
			# Be extra careful about order of operations for the test
			rho_hand_calc = rho_0
			rho_hand_calc /= r/r_scale
			rho_hand_calc /= (1+r/r_scale)**2
			self.assertAlmostEqual(integrator.nfw_rho(r_scale,
				rho_0,pos_nfw,pos),rho_hand_calc)

	def test_leapfrog_int_nfw_f_dyn(self):
		# There is no code for us to compare to here, so we can instead to some
		# sanity checks on limiting cases. First, let's input some
		# configurations that are equivalent to just an NFW and make sure
		# that's the result we get.

		# Set the problem up for circular motion.
		rho_0 = 1
		r_scale = 1
		G = 0.8962419740798497
		dt = 0.01
		num_dt = int(1e4)

		pos_nfw_array = np.tile(np.zeros(3,dtype=np.float64),(num_dt,1))
		save_pos_dyn_f = np.zeros((num_dt+1,3))
		save_vel_dyn_f = np.zeros((num_dt+1,3))
		save_pos = np.zeros((num_dt+1,3))
		save_vel = np.zeros((num_dt+1,3))
		rho_0_array = rho_0*np.ones(num_dt,dtype=np.float64)
		r_scale_array = r_scale*np.ones(num_dt,dtype=np.float64)
		r_init = 2
		pos_init = np.array([r_init,0,0],dtype=np.float64)
		M_r = 4*np.pi*rho_0*r_scale**3*(np.log((r_scale+r_init)/r_scale)+
			r_scale/(r_scale+r_init) - 1)
		v_r = np.sqrt(G*M_r/r_init)
		vel_init = np.array([0,v_r,0],dtype=np.float64)
		m_nfw_array = M_r*np.ones(num_dt,dtype=np.float64)
		v_max_nfw_array = np.ones(num_dt,dtype=np.float64)*np.sqrt(
			G*M_r/r_scale)
		rho_vir = 0.001

		# These are the two parameters we'll play with. First we'll give the
		# subhalo no mass. In that regime there should be no dynamical
		# friction
		m_sub = M_r/1e20
		v_max_nfw_array *= 1

		integrator.leapfrog_int_nfw_f_dyn(pos_init,vel_init,m_sub,rho_0_array,
			r_scale_array,pos_nfw_array,m_nfw_array,v_max_nfw_array,
			rho_vir,dt,save_pos_dyn_f,save_vel_dyn_f)

		# Reset the initial velocity and position
		pos_init = np.array([r_init,0,0],dtype=np.float64)
		vel_init = np.array([0,v_r,0],dtype=np.float64)
		integrator.leapfrog_int_nfw(pos_init,vel_init,rho_0_array,
			r_scale_array,pos_nfw_array,dt,save_pos,save_vel)
		np.testing.assert_almost_equal(save_pos,save_pos_dyn_f)

		# Now increase the subhalo mass but increase the velocity dispersion
		# of the NFW (and therefore make dynamical friction almost impossible)
		m_sub = M_r/1e2
		v_max_nfw_array *= 1e4

		pos_init = np.array([r_init,0,0],dtype=np.float64)
		vel_init = np.array([0,v_r,0],dtype=np.float64)
		integrator.leapfrog_int_nfw_f_dyn(pos_init,vel_init,m_sub,rho_0_array,
			r_scale_array,pos_nfw_array,m_nfw_array,v_max_nfw_array,
			rho_vir,dt,save_pos_dyn_f,save_vel_dyn_f)
		np.testing.assert_almost_equal(save_pos,save_pos_dyn_f)

		# Now make sure that dynamical friction behaves as we want it to.
		v_max_nfw_array /= 1e4

		pos_init = np.array([r_init,0,0],dtype=np.float64)
		vel_init = np.array([0,v_r,0],dtype=np.float64)
		integrator.leapfrog_int_nfw_f_dyn(pos_init,vel_init,m_sub,rho_0_array,
			r_scale_array,pos_nfw_array,m_nfw_array,v_max_nfw_array,
			rho_vir,dt,save_pos_dyn_f,save_vel_dyn_f)

		# Make sure the radius is smaller and decreasing due to the friction
		r_f_dyn = np.sqrt(np.sum(np.square(save_pos_dyn_f),axis=-1))
		r_nfw = np.sqrt(np.sum(np.square(save_pos),axis=-1))
		self.assertEqual(np.sum(r_f_dyn>r_nfw),0)
