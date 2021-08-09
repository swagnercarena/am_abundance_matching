#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <inttypes.h>
#include "distance.h"
#include "masses.h"
#include "universe_time.h"
#include "orphans.h"
#include "universal_constants.h"

#undef Gc
#define Gc 4.39877036e-15  /* In Mpc^2 / Myr / Msun * km/s */
#define PERIODIC 1

extern double box_size; /* Automatically set; in Mpc/h */
extern double Om, Ol, h0; /* Hubble constant */

double dynamical_time(double a);
double vir_density(double a);

/* From http://arxiv.org/abs/1403.6827 */
double jiang_mass_loss_factor(double a, double m, double M) {
  return -0.93*pow(m/M, 0.07)/(dynamical_time(a)*M_PI_2);
}

// Given a mass and a scale factor, return the virial radius
double m_to_rvir(double m, double a) { //In kpc/h
  float mean_density = CRITICAL_DENSITY*Om; //(Msun/h) / (comoving Mpc/h)^3
  // get the virial density at a specific scale factor
  float vd = vir_density(a)*mean_density;
  return (cbrt(m / (4.0*M_PI*vd/3.0)) * 1000.0);
}

// Evolve the orphan h between the host at p1 and the host at p2.
/*
Keep a running list of all choices being made
1) Position of host is linear interpolation of all three coordinates
2) Mass of host is linearly interpolated between time steps.
3) Using the scale radius fit from Rockstar and it's being cubically interpolated.
4) Apply force softening proportional to virial radius
*/
void evolve_orphan_halo(struct halo *h, struct halo *p1, struct halo *p2)
{
  int64_t i=0;
  float dt = (scale_to_years(p2->scale) - scale_to_years(p1->scale))/1.0e6 / (double)NUM_STEPS; //In Myr
  // av_a is the average scale factor
  float av_a = (p1->scale+p2->scale)/2.0;
  // I think this is getting the fundamental units of velocity we'll be using. Looks like calculations will be done
  // in comoving space.
  float vel_dt = dt * 1.02268944e-6 * h0 / av_a; //1 km/s to comoving Mpc/Myr/h
  float vir_conv_factor = pow(SOFTENING_LENGTH*1.0e-3, 2); //1e-3 is for converting Rvir from kpc/h to Mpc/h
  float max_dist = (PERIODIC) ? (box_size / 2.0) : (2.0*box_size);

  // Easy enough, just converting to H(t) using the scale factor
  float H = h0*100.0 * sqrt(Om/(av_a*av_a*av_a)+Ol); //Hubble flow in km/s/(Mpc)
  // So this is the hubble drag force. I think he's multiplying by km/Mpc and then by s/Myr.
  // That gets me the right units, except why the 2 and why the scale factor?
  float h_drag = -2.0 * H * av_a * 1.02269032e-6; //km/s/(Mpc) to km/s/Myr / (km/s) = 1/Myr

  // 1 km/s / Myr to comoving Mpc/Myr^2/h 
  float acc_dt2 = (dt*dt / 2.0) * 1.02268944e-6 * h0 / av_a;
  float conv_const = av_a*av_a / h0 / h0;

  // Okay here we initialize the acceleration and calculate mass loss
  double acc[3] = {0};
  double mf1 = calculate_mass_factor(p1->mvir, p1->rvir, p1->rs);
  double mf2 = calculate_mass_factor(p2->mvir, p2->rvir, p2->rs);

  /* Mass loss depends only weakly on Mhost */
  double jmlf = 1e6*jiang_mass_loss_factor(av_a, h->mvir, 0.5*(p1->mvir+p2->mvir));

  // This is the loop that actually does the leapfrog integration
  for (i=0; i<NUM_STEPS+1; i++) {
    if (i != 0) { //If we're not at the first step.
      for (int64_t j=0; j<3; j++)
	h->vel[j] += acc[j]*dt*0.5;
    }
    // The bit above is just doing the kick operation if you're not at the first
    // step.
    
    // If we're done, save the info and return. That makes sense.
    if (i==NUM_STEPS) {
      h->scale = p2->scale;
      h->rvir = m_to_rvir(h->mvir, p2->scale);
      return; //Return if this is the final step.
    }
    
    // p is the interpolated host halo that will contain all of the information we'll
    // use for the host in the integration.
    struct halo p = *p1;
    double f = (double)(i+0.5) / (double)NUM_STEPS;
    
    // Position of p is just linear interpolation
    for (int64_t j=0; j<3; j++)
      p.pos[j] += f*(p2->pos[j]-p1->pos[j]);
    
    // Mass is linear and scale radius is cubic
    double mf = mf1 + f*(mf2-mf1);
    double rs = cbrt((1.0-f)*pow(p1->rs, 3) + f*pow(p2->rs, 3));
    float inv_rs = 1000.0/rs;
    
    float r,m, dx, dy, dz, r3, rsoft2, accel;
    
    // Get the distance with boundary conditions
#define DIST(a,b) { a = p.b - h->b; if (a > max_dist) a-=box_size; else if (a < -max_dist) a+=box_size;  }
    DIST(dx,pos[0]);
    DIST(dy,pos[1]);
    DIST(dz,pos[2]);
#undef DIST
    
    // This is the dot product of velocity and displacement
    double vdotr = dx*h->vel[0] + dy*h->vel[1] + dz*h->vel[2];
    
    // radius
    r = sqrtf(dx*dx + dy*dy + dz*dz); // in comoving Mpc/h
    
    // This is the enclosed mass at this radius for the NFW profile we're
    // assuming.
    m = mf*ff_cached(r*inv_rs); //In Msun
    
    //Apply force softening:
    // The complicated vir_conv_factor at the end is applying force softening.
    // This is one difference between my code and universemachine.
    rsoft2 = r*r + h->rvir*h->rvir*vir_conv_factor;
    r3 = rsoft2 * r * conv_const; //in (real Mpc)^2 * (comoving Mpc/h)
    accel = r ? Gc*m/r3 : 0; //In km/s / Myr / (comoving Mpc/h)
    
    // So the acceleration is the classical newtonian value plus the hubble
    // drag. Somehow though, the drag is proportional to velocity. This could be
    // really important, since now the effect of the hubble constant is to
    // slow down particles which is what we need. Everything else from here on
    // out looks like standard leapfrog with comoving coordinates take into
    // account.
    acc[0] = accel*dx + h_drag*h->vel[0];      //In km/s / Myr
    acc[1] = accel*dy + h_drag*h->vel[1];
    acc[2] = accel*dz + h_drag*h->vel[2];
    
    if (vdotr < 0) // rhat points toward host halo
      h->mvir += 2.0*jmlf*dt*h->mvir;
    
    if (i != 0) { //If not at the first step
      for (int64_t j=0; j<3; j++) h->vel[j] += acc[j]*dt*0.5;
    }
    
    //Calculate new positions
    for (int64_t j=0; j<3; j++) {
      h->pos[j] += h->vel[j]*vel_dt + acc[j]*acc_dt2;
      if (PERIODIC) {
	if (h->pos[j] < 0) h->pos[j] += box_size;
	if (h->pos[j] > box_size) h->pos[j] -= box_size;
      }
    }
  }
}
    