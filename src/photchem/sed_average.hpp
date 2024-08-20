#ifndef PHOTCHEM_SED_AVERAGE_HPP_
#define PHOTCHEM_SED_AVERAGE_HPP_
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file sed_average.hpp
//! \brief definitions for SEDAverage types
//========================================================================================

// Athena++ headers
#include "../athena.hpp"

constexpr int NFREQ_SIMPLE = NFREQ_RAYT;

// Use POD type for SED-averaged quantities. We may need more sophisticated approach using
// CRTP and/or mixin. This will be updated in the future.

template <class ...Bases>
struct SEDAverageBase : Bases... {
};

template<int n>
struct SEDAverageHIAndDust {
  // Spatially constant cross-sections per H for setting opacity chi used in ray tracing
  // Need more generalized approach for full photochemistry
  // Mean photon energy in eV
  Real hnu[n];         // NOLINT (runtime/arrays)
  // Dust absorption cross section [area/H]
  Real sigma_d[n];     // NOLINT (runtime/arrays)
  // Dust absorption cross section  [area/mass]
  Real kappa_d[n];     // NOLINT (runtime/arrays)
  // HI photoionization cross section [area/H]
  Real sigma_pi_hi[n]; // NOLINT (runtime/arrays)
  // HI photoionization cross section [area/mass]
  Real kappa_pi_hi[n]; // NOLINT (runtime/arrays)
};

template<int n>
struct SEDAverageH2 {
  // Spatially constant cross-sections per H for setting opacity chi used in ray tracing
  // TODO(JGKIM): Need more generalized approach for full photochemistry
  // TODO(JGKIM): Need also connect with population synthesis
  // H2 photoionization cross section [area/H]
  Real sigma_pi_h2[n]; // NOLINT (runtime/arrays)
  // H2 photoionization cross section [area/mass]
  Real kappa_pi_h2[n]; // NOLINT (runtime/arrays)
};

using SEDAverageSimple = SEDAverageBase<SEDAverageHIAndDust<NFREQ_SIMPLE>>;

#endif  // PHOTCHEM_SED_AVERAGE_HPP_
