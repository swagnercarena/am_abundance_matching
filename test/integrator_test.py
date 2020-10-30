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
		rho_0 = 1e10
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
		rho_0 = 1e11
		r_scale = 1
		G = 0.8962419740798497

		dt = 0.0001
		num_dt = int(1e5)

		pos_nfw_array = np.tile(np.zeros(3,dtype=np.float64),(num_dt,1))
		save_pos = np.zeros((num_dt+1,3))
		save_vel = np.zeros((num_dt+1,3))

		r_init = 20
		pos_init = np.array([r_init,0,0],dtype=np.float64)

		M_r = 4*np.pi*rho_0*r_scale**3*(np.log((r_scale+r_init)/r_scale)+
			r_scale/(r_scale+r_init) - 1)
		v_r = np.sqrt(G*M_r/r_init)
		vel_init = np.array([0,v_r,0],dtype=np.float64)
		period = 2*np.pi*np.sqrt(r_init**3/(G*M_r))
		# Get the final completed period
		last_period = int(int(num_dt*dt/period)*period//dt)

		integrator.leapfrog_int_nfw(pos_init,vel_init,rho_0,r_scale,
			pos_nfw_array,dt,save_pos,save_vel)
		# Check that it returns to the original position once per period.
		np.testing.assert_almost_equal(save_pos[0,:],save_pos[last_period,:],
			decimal=3)

		# Repeat the test with a moving NFW
		pos_nfw_array = np.tile(np.zeros(3,dtype=np.float64),(num_dt,1))
		for pi, pos in enumerate(pos_nfw_array):
			pos[0] += pi*dt

		pos_init = np.array([r_init,0,0],dtype=np.float64)
		vel_init = np.array([1,v_r,0],dtype=np.float64)
		integrator.leapfrog_int_nfw(pos_init,vel_init,rho_0,r_scale,
			pos_nfw_array,dt,save_pos,save_vel)

		np.testing.assert_almost_equal(save_pos[0,:],
			save_pos[last_period,:]-pos_nfw_array[last_period,:],
			decimal=3)

	def test_leapfrog_nfw_galpy(self):
		# Compare a more complicated set of initial conditions to the galpy
		# outputs (which are slow but known to work).

		# Set the parameters for a slightly offset circular orbit
		rho_0 = 1e11
		r_scale = 1
		G = 0.8962419740798497

		dt = 0.001
		num_dt = int(3e4)

		pos_nfw_array = np.tile(np.zeros(3,dtype=np.float64),(num_dt,1))
		save_pos = np.zeros((num_dt+1,3))
		save_vel = np.zeros((num_dt+1,3))

		r_init = 20
		pos_init = np.array([r_init,0,0],dtype=np.float64)

		M_r = 4*np.pi*rho_0*r_scale**3*(np.log((r_scale+r_init)/r_scale)+
			r_scale/(r_scale+r_init) - 1)
		v_r = np.sqrt(G*M_r/r_init)
		v_kick = 1e-1
		vel_init = np.array([v_kick,v_r,0],dtype=np.float64)

		integrator.leapfrog_int_nfw(pos_init,vel_init,rho_0,r_scale,
			pos_nfw_array,dt,save_pos,save_vel)

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
		integrator.leapfrog_int_nfw(pos_init,vel_init,rho_0,r_scale,
			pos_nfw_array,dt,save_pos,save_vel)

		# Do the same for galpy
		o=Orbit([r_init,v_kick,v_r,0,v_kick,0])
		nfw = NFWPotential(a=r_scale,amp=rho_0*G*4*np.pi*r_scale**3)
		ts = np.linspace(0,dt*num_dt,num_dt+1)
		o.integrate(ts,nfw,method='leapfrog',dt=dt)

		np.testing.assert_almost_equal(o.x(ts),save_pos[:,0])
		np.testing.assert_almost_equal(o.y(ts),save_pos[:,1])
		np.testing.assert_almost_equal(o.z(ts),save_pos[:,2])

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
		x = 1/(np.sqrt(2)*sigma)
		x_factor = (math.erf(x)-2*x/np.sqrt(np.pi)*np.exp(-x**2))
		ln_lamba = 4

		hand_calc = -4*np.pi*G**2*m_sub/3*ln_lamba*x_factor*v/np.sqrt(3)
		np.testing.assert_almost_equal(hand_calc,
			integrator.calc_neg_grad_f_dyn(v,rho,m_sub,m_host,sigma))
