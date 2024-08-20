#ifndef PHOTCHEM_PHOTCHEM_HPP_
#define PHOTCHEM_PHOTCHEM_HPP_
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file photchem.hpp
//! \brief definitions for Photochemistry class

// C headers

// C++ headers
#include <cmath>      // pow()

// Athena++ headers
#include "../athena.hpp"
#include "../globals.hpp"
#include "../units/units.hpp" // Constants
// needed for NFREQ (TODO: make it work for more general cases)
#include "../ray_tracing/ray_tracing.hpp"
#include "sed_average.hpp"

class PhotochemistryDriver;

enum class PhotochemistryMode {simple};

enum SpeciesIndex {IHI=0};
enum FrequencyIndexSimple {ILyC=0, IFUV=1};

class Photochemistry {
  friend PhotochemistryDriver;

 public:
  Photochemistry(MeshBlock *pmy_block, ParameterInput *pin);
  virtual ~Photochemistry() {}

  virtual void UpdateSourceTermsOperatorSplit() = 0;

  // Inlined function (if we are using CRTP)
  // Real GetMeanPhotonEnergy(int ifr) const {
  //   return static_cast<T*>(this)->sedavg_.hnu[ifr];
  // }

 protected:
  MeshBlock *pmb;
  Units *punit;
  PhotochemistryMode mode_;
  AthenaArray<Real> rad_mom_;
  AthenaArray<Real> rad_mom0_fshld_;

 private:
  void SetRadiationMoments();
  virtual void SetCrossSectionsAndPhotonEnergies(ParameterInput *pin) = 0;
};

//////////////////////////////
// Two-temperature isothermal
//////////////////////////////
class PhotochemistrySimple : public Photochemistry {
  friend PhotochemistryDriver;

 public:
  PhotochemistrySimple(MeshBlock *pmb, ParameterInput *pin);
  ~PhotochemistrySimple();

  static Real PhotochemistryTimeStep(MeshBlock *pmb);
  void ApplyFloorAndCeiling();
  void CalculateRates();

  // TODO(JGKIM) : need to improve this in the future
  // Inlined function
  Real GetMeanPhotonEnergy(int ifr) const { return sedavg_.hnu[ifr]; }
  static void UpdateSourceTerms(MeshBlock *pmb, const Real t, const Real dt,
                                const AthenaArray<Real> &prim,
                                const AthenaArray<Real> &prim_scalar,
                                const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
                                AthenaArray<Real> &cons_scalar);

  void UpdateSourceTermsOperatorSplit() override;

  // Inlined functions
  Real CaseBRecCoefficient(const Real tgas);
  Real GetTemperature(const Real x_hi);
  Real GetPressureCGS(const Real n_h, const Real tgas, const Real x_hi);

  static void UpdateRayTracingOpacitySimple(MeshBlock *pmb, AthenaArray<Real> &chi);

  // For saving passive scalar source terms. Allocation made in pgen as user output
  // variables
  bool bookkeeping;
  AthenaArray<Real> rho_hi_dot, x_hi_before;

  AthenaArray<Real> trec_inv_;
  AthenaArray<Real> tionrec_inv_;
  AthenaArray<Real> x_hi_eq_;

 private:
  static constexpr Real x_hi_floor_ = 1e-8;
  static constexpr Real x_he_ = 0.1;
  const Real tgas_hi_, tgas_hii_;
  Real f_dt_rad_;
  Real min_dt_;
  Real max_dx_hi_eq_;
  SEDAverageSimple sedavg_;

  void SetRadiationMomentWithShielding();
  void SetCrossSectionsAndPhotonEnergies(ParameterInput *pin) override;
};


class PhotochemistryDriver {
 public:
  PhotochemistryDriver(Mesh *pm, ParameterInput *pin);
  ~PhotochemistryDriver();

  Photochemistry* CreatePhotochemistry(MeshBlock *pmb, ParameterInput *pin);
  void UpdateRatesAndTimeStep();

  Units *punit;
  PhotochemistryMode mode;

  const bool flag_op_split;
  // Apply radiation pressure force (kappa rho flux /c times dt)
  const bool flag_rad_force;
  // Update time step size of the main time integrator
  const bool flag_update_dt_main;

 private:
  Mesh *pmy_mesh_;

  void SetMode(ParameterInput *pin);
};

inline Real PhotochemistrySimple::CaseBRecCoefficient(const Real tgas) {
  return 2.59e-13*std::pow(tgas*1e-4,-0.7);
}

inline Real PhotochemistrySimple::GetTemperature(const Real x_hi) {
  return tgas_hii_ - x_hi/(2.0 - x_hi)*(tgas_hii_ - tgas_hi_);
}

inline Real PhotochemistrySimple::GetPressureCGS(const Real n_h, const Real tgas,
                                                 const Real x_hi) {
  return (2.0 + x_he_ - x_hi)*n_h*Constants::k_boltzmann_cgs*tgas;
}


#endif  // PHOTCHEM_PHOTCHEM_HPP_
