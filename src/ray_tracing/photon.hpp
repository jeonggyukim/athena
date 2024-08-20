#ifndef RAY_TRACING_PHOTON_HPP_
#define RAY_TRACING_PHOTON_HPP_
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file photon.hpp
//! \brief definitions for photon POD types
//========================================================================================

// Athena++ headers
#include "../athena.hpp"

template <class ...Bases>
struct PhotonBase : Bases... {
};

template<int n>
struct PhotonSimple {
  int lev;          //!> HEALPix level
  int levptr;       //!> HEALPix pixel number
  int mb_idx;       //!> local MeshBlock index
  int icell, jcell, kcell; //!> cell indices in meshblock

  Real x1_src, x2_src, x3_src; //!> source position
  Real n1, n2, n3; //!> unit vector specifying ray direction
  Real dist;       //!> distance from source to current position
  Real lum[n]; //!> luminosity  // NOLINT (runtime/arrays)
  Real tau[n]; //!> optical depth from source // NOLINT (runtime/arrays)
};

using Photon = PhotonBase<PhotonSimple<NFREQ_RAYT>>;

#endif  // RAY_TRACING_PHOTON_HPP_
