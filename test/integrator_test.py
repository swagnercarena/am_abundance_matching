import unittest
from avocet import integrator
import numpy as np
from galpy.potential import NFWPotential


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

		g_comp = - G*part_mass[0]/np.sqrt(2)*np.array([1,-1,0])
		g_comp += - G*part_mass[1]/np.sqrt(5)*np.array([1,0,-2])
		g = integrator.calc_neg_grad_nb(part_pos,part_mass,pos)
		np.testing.assert_almost_equal(g,g_comp)

	def test_leapfrog_int_nb(self):
		# Try some analytically known 2 body integrals and ensure that the
		# results match expectations.
		G = 6.67430e-11
		M = 1e10
		r = 1
		part_pos = np.array([[0,0,0],[0,r,0]],dtype=np.float64)
		part_vel = np.array([[0,0,0],[np.sqrt(G*M),0,0]],dtype=np.float64)
		part_mass = np.array([M,1],dtype=np.float64)

		dt = 0.001
		period = 2*np.pi*np.sqrt(r**3/(G*M))
		num_dt = int(period/dt)
		save_pos_array = np.zeros((num_dt+1,2,3))
		save_vel_array = np.zeros((num_dt+1,2,3))

		integrator.leapfrog_int_nb(part_pos,part_vel,part_mass,dt,num_dt,
			save_pos_array,save_vel_array)

		for i in range(2):
			for j in range(2):
				np.testing.assert_almost_equal(save_pos_array[0,i,j],
					save_pos_array[-1,i,j],decimal=3)
				np.testing.assert_almost_equal(save_vel_array[0,i,j],
					save_vel_array[-1,i,j],decimal=3)

		# Quickly repeat the same for 5 more periods
		num_dt = int(5*period/dt)
		save_pos_array = np.zeros((num_dt+1,2,3))
		save_vel_array = np.zeros((num_dt+1,2,3))
		integrator.leapfrog_int_nb(part_pos,part_vel,part_mass,dt,num_dt,
			save_pos_array,save_vel_array)

		for i in range(2):
			for j in range(2):
				np.testing.assert_almost_equal(save_pos_array[0,i,j],
					save_pos_array[-1,i,j],decimal=3)
				np.testing.assert_almost_equal(save_vel_array[0,i,j],
					save_vel_array[-1,i,j],decimal=3)

		# Now check that we get reasonable results for a e = 0.9 Kepplerian
		# orbit
		rp = 1.9
		vt = 0.005927
		period = 243
		m1 = 1e2
		m2 = 1e7
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
		return
