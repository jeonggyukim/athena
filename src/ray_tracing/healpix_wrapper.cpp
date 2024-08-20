//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file healpix_wrapper.cpp
//! \brief Implementation of functions in class HEALPixWrapper
//========================================================================================

// C headers

// C++ headers
#include <cmath>      // sin, cos, sqrt

// Athena++ headers
#include "../athena.hpp"
#include "../parameter_input.hpp"
#include "ray_tracing.hpp"

// healpix.cpp
void pix2ang_nest_z_phi(int nside, int pix, Real *z, Real *phi);

//--------------------------------------------------------------------------------------
//! \fn HEALPixWrapper::HEALPixWrapper()
//! \brief Constructs a HEALPixWrapper instance.

HEALPixWrapper::HEALPixWrapper(bool rotate_rays, int rseed) :
  rotate_rays_(rotate_rays), rseed_(rseed), udist_(0.0,TWO_PI) {
  if (rotate_rays_) {
    // std::random_device device;
    // rseed_ = static_cast<std::int64_t>(device());
    rng_generator_.seed(rseed_);
  }
}

//--------------------------------------------------------------------------------------
//! \fn HEALPixWrapper::HEALPixWrapper()
//! \brief Destroys a HEALPixWrapper instance.

HEALPixWrapper::~HEALPixWrapper() {
}

//--------------------------------------------------------------------------------------
//! \fn HEALPixWrapper::Pix2VecNestWithoutRotation()
//! \brief Calculates a vector pointing in the direction of the pixel center.

void HEALPixWrapper::Pix2VecNestWithoutRotation(int nside, int ipix, Real *vec) {
  Real z, phi, stheta;
  pix2ang_nest_z_phi(nside,ipix,&z,&phi);
  stheta = std::sqrt((1.-z)*(1.+z));
  vec[0] = stheta*std::cos(phi);
  vec[1] = stheta*std::sin(phi);
  vec[2] = z;
}

//--------------------------------------------------------------------------------------
//! \fn HEALPixWrapper::Pix2VecNestWithRotation()
//! \brief Calculates a rotated unit HEALPix vector.

void HEALPixWrapper::Pix2VecNestWithRotation(int nside, int ipix, Real *vec) {
  Real z, phi, stheta, vec_[3];
  pix2ang_nest_z_phi(nside,ipix,&z,&phi);
  stheta = std::sqrt((1.-z)*(1.+z));
  // Rotation around z-axis can be performed by just adding phi_euler to phi
  vec_[0] = stheta*std::cos(phi + phi_euler_);
  vec_[1] = stheta*std::sin(phi + phi_euler_);
  vec_[2] = z;
  // Rotation around x' followed by around z''
  vec[0] = rotation_[0][0]*vec_[0] + rotation_[0][1]*vec_[1] +
    rotation_[0][2]*vec_[2];
  vec[1] = rotation_[1][0]*vec_[0] + rotation_[1][1]*vec_[1] +
    rotation_[1][2]*vec_[2];
  vec[2] = rotation_[2][0]*vec_[0] + rotation_[2][1]*vec_[1] +
    rotation_[2][2]*vec_[2];
  return;
}


//--------------------------------------------------------------------------------------
//! \fn HEALPixWrapper::UpdateHEALPixRotationMatrix()
//! \brief Calculates a rotated unit HEALPix vector.

void HEALPixWrapper::UpdateHEALPixRotationMatrix() {
  phi_euler_ = udist_(rng_generator_);
  theta_euler_ = udist_(rng_generator_);
  psi_euler_ = udist_(rng_generator_);

  Real cos_theta = std::cos(theta_euler_);
  Real sin_theta = std::sin(theta_euler_);
  Real cos_psi = std::cos(psi_euler_);
  Real sin_psi = std::sin(psi_euler_);

  rotation_[0][0] =  cos_psi;
  rotation_[0][1] = -sin_psi*cos_theta;
  rotation_[0][2] =  sin_psi*sin_theta;

  rotation_[1][0] =  sin_psi;
  rotation_[1][1] =  cos_psi*cos_theta;
  rotation_[1][2] = -cos_psi*sin_theta;

  rotation_[2][0] =  0.0;
  rotation_[2][1] =  sin_theta;
  rotation_[2][2] =  cos_theta;

  return;
}
