//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file photchem_simple.cpp
//! \brief Implementation of functions in class PhotochemistrySimple
//========================================================================================

// C headers

// C++ headers
#include <cmath>

// Athena++ headers
#include "../athena.hpp"
#include "../bvals/bvals.hpp"
#include "../eos/eos.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"
#include "../ray_tracing/ray_tracing.hpp"
#include "../scalars/scalars.hpp"
#include "../units/units.hpp"

#include "photchem.hpp"

constexpr Real PhotochemistrySimple::x_hi_floor_;

//--------------------------------------------------------------------------------------
//! \fn PhotochemistrySimple::PhotochemistrySimple(MeshBlock *pmb, ParameterInput *pin)
//! \brief Constructs a PhotochemistrySimple instance.

PhotochemistrySimple::PhotochemistrySimple(MeshBlock *pmy_block, ParameterInput *pin) :
  Photochemistry(pmy_block, pin),
  bookkeeping(pin->GetOrAddBoolean("photchem", "bookkeeping", false)),
  tgas_hi_(pin->GetOrAddReal("photchem", "tgas_HI", 1.0e2)),
  tgas_hii_(pin->GetOrAddReal("photchem", "tgas_HII", 8.0e3)) {
  SetCrossSectionsAndPhotonEnergies(pin);

  f_dt_rad_ = pin->GetOrAddReal("photchem", "f_dt_rad", 0.1);
  if (f_dt_rad_ >= 1.0) {
    std::stringstream msg;
    msg << "### FATAL ERROR in PhotochemistrySimple" << std::endl
        << "f_dt_rad = " << f_dt_rad_ << " , but it should be less than 1" << std::endl;
    ATHENA_ERROR(msg);
  }

  const int ncells1 = pmb->ncells1, ncells2 = pmb->ncells2, ncells3 = pmb->ncells3;
  trec_inv_.NewAthenaArray(ncells3,ncells2,ncells1);
  tionrec_inv_.NewAthenaArray(ncells3,ncells2,ncells1);
  x_hi_eq_.NewAthenaArray(ncells3,ncells2,ncells1);
  if (pmb->pmy_mesh->ray_tracing)
    pmb->prayt->EnrollOpacityFunction(UpdateRayTracingOpacitySimple);
}


//--------------------------------------------------------------------------------------
//! \fn PhotochemistrySimple::~PhotochemistrySimple()
//! \brief Destructs a PhotochemistrySimple instance.

PhotochemistrySimple::~PhotochemistrySimple() {
}


//--------------------------------------------------------------------------------------
//! \fn PhotochemistrySimple::PhotochemistryTimeStep()
//! \brief Returns a timestep size for photochemistry update.

Real PhotochemistrySimple::PhotochemistryTimeStep(MeshBlock *pmb) {
  PhotochemistrySimple *ppc = static_cast<PhotochemistrySimple*>(pmb->pphotchem);
  return ppc->min_dt_;
}


//--------------------------------------------------------------------------------------
//! \fn PhotochemistrySimple::SetCrossSectionsAndPhotonEnergies()
//! \brief Initialize SED-averaged cross sections and photon energies, etc.

void PhotochemistrySimple::SetCrossSectionsAndPhotonEnergies(ParameterInput *pin) {
  // TODO(JGKIM): add option to read opacity from pre-existing tables or to calculate
  // cross sections on-the-fly. Eventually, this function should probably belong to SED
  // class.

  // Read cross sections from input
  Real sigma_to_kappa = 1.0/(1.4*Constants::hydrogen_mass_cgs*punit->gram_code);
  std::string base_hnu = "hnu";
  std::string base_sigma_d = "sigma_d";
  std::string base_sigma_pi_hi = "sigma_pi_HI";
  for (int ifr=0; ifr<NFREQ_SIMPLE; ++ifr) {
    std::string str_hnu = base_hnu + "[" + std::to_string(ifr) + "]";
    std::string str_sigma_pi_hi = base_sigma_pi_hi + "[" + std::to_string(ifr) + "]";
    std::string str_sigma_d = base_sigma_d + "[" + std::to_string(ifr) + "]";
    sedavg_.hnu[ifr] = pin->GetReal("photchem", str_hnu)*punit->electron_volt_code;
    sedavg_.sigma_d[ifr] = pin->GetReal("photchem", str_sigma_d)*SQR(punit->cm_code);
    sedavg_.kappa_d[ifr] = sedavg_.sigma_d[ifr]*sigma_to_kappa;
    sedavg_.sigma_pi_hi[ifr] = pin->GetReal("photchem", str_sigma_pi_hi)*SQR(punit->cm_code);
    sedavg_.kappa_pi_hi[ifr] = sedavg_.sigma_pi_hi[ifr]*sigma_to_kappa;
  }
  return;
}


//--------------------------------------------------------------------------------------
//! \fn PhotochemistrySimple::UpdateRayTracingOpacitySimple()
//! \brief Updates an opacity array used for ray tracing.

void PhotochemistrySimple::UpdateRayTracingOpacitySimple(MeshBlock *pmb,
                                                         AthenaArray<Real> &chi) {
  const int ks=pmb->ks, ke=pmb->ke;
  const int js=pmb->js, je=pmb->je;
  const int is=pmb->is, ie=pmb->ie;
  Hydro *phydro = pmb->phydro;
  PassiveScalars *pscalars = pmb->pscalars;
  PhotochemistrySimple *ppc = static_cast<PhotochemistrySimple*>(pmb->pphotchem);

  // Apply floor and ceilng first
  ppc->ApplyFloorAndCeiling();
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
#pragma omp simd
      for (int i=is; i<=ie; ++i) {
        const Real rho = phydro->u(IDN,k,j,i);
        const Real rho_hi = pscalars->s(IHI,k,j,i);
        // NOTE: Not needed if floor/ceiling is applied beforehand
        // rho_hi = std::max(x_hi_floor_*rho, rho_hi);
        // rho_hi = std::min(rho, rho_hi);
        for (int ifr=0; ifr<NFREQ_SIMPLE; ++ifr) {
          chi(k,j,i,ifr) = (rho*ppc->sedavg_.kappa_d[ifr] +
                            rho_hi*ppc->sedavg_.kappa_pi_hi[ifr]);
        }
      }
    }
  }
  return;
}


//--------------------------------------------------------------------------------------
//! \fn PhotochemistrySimple::ApplyFloorAndCeiling()
//! \brief Applies floor and ceiling to a passive scalar.

void PhotochemistrySimple::ApplyFloorAndCeiling() {
  // For now, this function is called before calculating ray tracing opacity and before
  // doing rate calculation.
  const int ks=pmb->ks, ke=pmb->ke;
  const int js=pmb->js, je=pmb->je;
  const int is=pmb->is, ie=pmb->ie;
  Hydro *phydro = pmb->phydro;
  PassiveScalars *pscalars = pmb->pscalars;

  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
#pragma omp simd
      for (int i=is; i<=ie; ++i) {
        const Real rho = phydro->u(IDN,k,j,i);
        const Real rho_hi_floor = rho*x_hi_floor_;
        Real &rho_hi = pscalars->s(IHI,k,j,i);
        rho_hi = std::max(rho_hi, rho_hi_floor);
        rho_hi = std::min(rho_hi, rho);
        pscalars->r(IHI,k,j,i) = rho_hi/rho;
      }
    }
  }
  return;
}


//--------------------------------------------------------------------------------------
//! \fn PhotochemistrySimple::CalculateRates()
//! \brief Calculates recombination and ionization timescales and dt_chem.

void PhotochemistrySimple::CalculateRates() {
  const int ks=pmb->ks, ke=pmb->ke;
  const int js=pmb->js, je=pmb->je;
  const int is=pmb->is, ie=pmb->ie;
  const Real rho_to_nh = punit->code_density_cgs/(1.4*Constants::hydrogen_mass_cgs);
  const Real code_time_cgs = punit->code_time_cgs;
  const Real f_dt_rad = f_dt_rad_;
  Real min_dt = std::numeric_limits<Real>::max();
  Real max_dx_hi_eq = 0.0;
  Hydro *phydro = pmb->phydro;
  PassiveScalars *pscalars = pmb->pscalars;
  RayTracing *prayt = pmb->prayt;

  // Loop over cells to compute rates and equilibrium fraction
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
#pragma omp simd
      for (int i=is; i<=ie; ++i) {
        const Real rho = phydro->u(IDN,k,j,i);
        const Real n_h = rho*rho_to_nh;
        const Real x_hi = pscalars->s(IHI,k,j,i)/rho;
        // NOTE: Not needed if floor/ceiling is applied beforehand
        // x_hi = std::max(x_hi_floor_, x_hi);
        // x_hi = std::min(x_hi,1.0);
        const Real n_e = (1.0 - x_hi)*n_h;
        Real zeta_pi_hi = 0.0;
        for (int ifr=0; ifr<NFREQ_SIMPLE; ++ifr) {
          zeta_pi_hi += sedavg_.sigma_pi_hi[ifr]*
            (rad_mom_(IER_RT,k,j,i,ifr)/sedavg_.hnu[ifr]);
        }
        const Real tgas = GetTemperature(x_hi);
        const Real alpha_b = CaseBRecCoefficient(tgas);

        // Inverse of timescales in code units
        trec_inv_(k,j,i) = alpha_b*n_e*code_time_cgs;
        tionrec_inv_(k,j,i) = trec_inv_(k,j,i) + zeta_pi_hi;
        if (tionrec_inv_(k,j,i) != 0.0) {
          x_hi_eq_(k,j,i) = trec_inv_(k,j,i) / tionrec_inv_(k,j,i);
        } else {
          x_hi_eq_(k,j,i) = 1.0;
        }
        max_dx_hi_eq = std::max(max_dx_hi_eq,
                                std::fabs(x_hi - x_hi_eq_(k,j,i)));
        const Real dx_hi_abs = std::fabs(x_hi - x_hi_eq_(k,j,i));
        if ((dx_hi_abs > f_dt_rad) && (tionrec_inv_(k,j,i) != 0.0)) {
          Real dt = -std::log(1.0 - f_dt_rad/dx_hi_abs)/tionrec_inv_(k,j,i);
          min_dt = std::min(min_dt, dt);
        }
      }
    }
  }
  max_dx_hi_eq_ = max_dx_hi_eq;
  min_dt_ = min_dt;
  return;
}


//--------------------------------------------------------------------------------------
//! \fn PhotochemistrySimple::UpdateSourceTerms()
//! \brief Enrollable function for updateing photochemistry source terms.

void PhotochemistrySimple::UpdateSourceTerms(MeshBlock *pmb, const Real t, const Real dt,
       const AthenaArray<Real> &prim, const AthenaArray<Real> &prim_scalar,
       const AthenaArray<Real> &bcc, AthenaArray<Real> &cons,
       AthenaArray<Real> &cons_scalar) {
  // TODO(JGKIM): Update this function later if useful.
  const int ks=pmb->ks, ke=pmb->ke;
  const int js=pmb->js, je=pmb->je;
  const int is=pmb->is, ie=pmb->ie;
  const Real igm1 = 1.0/(pmb->peos->GetGamma() - 1.0);
  Units *punit = pmb->pmy_mesh->punit;
  const Real code_pressure_cgs_inv = 1.0/punit->code_pressure_cgs;
  const Real rho_to_nh = punit->code_density_cgs/(1.4*Constants::hydrogen_mass_cgs);
  PhotochemistrySimple *ppc = static_cast<PhotochemistrySimple*>(pmb->pphotchem);
  RayTracing *prayt = pmb->prayt;

  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
#pragma omp simd
      for (int i=is; i<=ie; ++i) {
        const Real rho = prim(IDN,k,j,i);
        const Real n_h = rho*rho_to_nh;
        const Real press = prim(IPR,k,j,i);
        const Real x_hi = prim_scalar(IHI,k,j,i);
        const Real n_e = (1.0 - x_hi)*n_h;
        // Analytic solution assuming constant ne, alphaB, and radiation field. (e.g., Eq.
        // 20 in KimJG et al. 2017)
        Real x_hi_next = x_hi + (ppc->x_hi_eq_(k,j,i) - x_hi)*
          (1.0 - std::exp(-dt*ppc->tionrec_inv_(k,j,i)));
        x_hi_next = std::max(x_hi_floor_, x_hi_next);
        cons_scalar(IHI,k,j,i) = n_h*x_hi_next;
        const Real tgas = ppc->GetTemperature(x_hi_next);
        const Real press_next = ppc->GetPressureCGS(n_h, tgas, x_hi_next)*
          code_pressure_cgs_inv;
        const Real delta_e = (press_next-press)*igm1;
        cons(IEN,k,j,i) += delta_e;
      }
    }
  }
  return;
}


//--------------------------------------------------------------------------------------
//! \fn PhotochemistrySimple::UpdateSourceTermsOperatorSplit()
//! \brief Updates abundances and internal energy.

void PhotochemistrySimple::UpdateSourceTermsOperatorSplit() {
  const int is = pmb->is, ie = pmb->ie;
  const int js = pmb->js, je = pmb->je;
  const int ks = pmb->ks, ke = pmb->ke;
  const Real igm1 = 1.0/(pmb->peos->GetGamma() - 1.0);
  const Real code_pressure_cgs_inv = 1.0/punit->code_pressure_cgs;
  const Real rho_to_nh = punit->code_density_cgs/(1.4*Constants::hydrogen_mass_cgs);
  const Real dt = pmb->pmy_mesh->dt;
  const Real dt_inv = 1/dt;
  Hydro *phydro = pmb->phydro;
  PassiveScalars *pscalars = pmb->pscalars;
  RayTracing *prayt = pmb->prayt;

  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
#pragma omp simd
      for (int i=is; i<=ie; ++i) {
        Real rho = phydro->u(IDN,k,j,i);
        Real n_h = rho*rho_to_nh;
        Real x_hi = pscalars->s(IHI,k,j,i)/rho;
        // Analytic solution assuming constant ne, alphaB, etc. (see Mellema et al. 2006)
        Real x_hi_next = x_hi + (x_hi_eq_(k,j,i) - x_hi)*
          (1.0 - std::exp(-dt*tionrec_inv_(k,j,i)));
        x_hi_next = std::max(x_hi_floor_, x_hi_next);
        x_hi_next = std::min(1.0, x_hi_next);
        pscalars->r(IHI,k,j,i) = x_hi_next;
        pscalars->s(IHI,k,j,i) = x_hi_next*rho;
        if (bookkeeping) {
          rho_hi_dot(k,j,i) = rho*(x_hi_next - x_hi)*dt_inv;
          x_hi_before(k,j,i) = x_hi;
        }
        Real tgas = GetTemperature(x_hi_next);
        Real press_next = GetPressureCGS(n_h, tgas, x_hi_next)*code_pressure_cgs_inv;
        // TODO(JGKIM): Is there a better way to do this?
        Real e_non_thermal = phydro->u(IEN,k,j,i) - phydro->w(IPR,k,j,i)*igm1;
        phydro->u(IEN,k,j,i) = press_next*igm1 + e_non_thermal;
        phydro->w(IPR,k,j,i) = press_next;
      }
    }
  }

  if (pmb->pmy_mesh->pphotchemd->flag_rad_force) {
    const Real c_inv = 1.0/punit->speed_of_light_code;
    // see appendix D in Kim, J.-G. et al. (2023)
    // critical velocities for thermal and non-thermal sputtering
    const Real cs_iso_crit = 110*punit->km_s_code; // T ~ 10^6 K
    const Real v_d_crit = 1e2*punit->km_s_code;
    // Critical value for drag function
    const Real g_s_crit = 50.0;
    // (approximate) radiation pressure efficiency factor (Eq. 24 in Draine 2011)
    const Real q_pr = 1.0;
    for (int k=ks; k<=ke; ++k) {
      for (int j=js; j<=je; ++j) {
#pragma omp simd
        for (int i=is; i<=ie; ++i) {
          const Real rho = phydro->u(IDN,k,j,i);
          const Real press = phydro->w(IPR,k,j,i);
          // NOTE: There is a slight inconsistency (a few percent level) of taking
          // cs_iso=sqrt(2kT/muH) and sqrt(2.1kT/muH) in other places. Assume that dust is
          // destroyed by thermal sputtering.
          const Real cs_iso = std::sqrt(press/rho);
          if (cs_iso > cs_iso_crit)
            continue;

          Real flux_x = 0.0, flux_y = 0.0, flux_z = 0.0;
          Real frad_x = 0.0, frad_y = 0.0, frad_z = 0.0;
          for (int ifr=0; ifr<NFREQ_SIMPLE; ++ifr) {
            flux_x += rad_mom_(IFX_RT,k,j,i,ifr);
            flux_y += rad_mom_(IFY_RT,k,j,i,ifr);
            flux_z += rad_mom_(IFZ_RT,k,j,i,ifr);
            frad_x += prayt->chi_rayt(k,j,i,ifr)*rad_mom_(IFX_RT,k,j,i,ifr)*c_inv;
            frad_y += prayt->chi_rayt(k,j,i,ifr)*rad_mom_(IFY_RT,k,j,i,ifr)*c_inv;
            frad_z += prayt->chi_rayt(k,j,i,ifr)*rad_mom_(IFZ_RT,k,j,i,ifr)*c_inv;
          }
          const Real flux_mag = std::sqrt(SQR(flux_x) + SQR(flux_y) + SQR(flux_z));
          const Real g_s = q_pr*flux_mag*c_inv/press;
          if (g_s > g_s_crit) {
            if (std::sqrt(g_s)*cs_iso > v_d_crit) {
              continue;
            }
          }

          // apply force
          const Real e_kin0 = 0.5*(SQR(phydro->u(IM1,k,j,i)) +
                                   SQR(phydro->u(IM2,k,j,i)) +
                                   SQR(phydro->u(IM3,k,j,i))) / rho;
          phydro->u(IM1,k,j,i) += frad_x*dt;
          phydro->u(IM2,k,j,i) += frad_y*dt;
          phydro->u(IM3,k,j,i) += frad_z*dt;
          const Real e_kin = 0.5*(SQR(phydro->u(IM1,k,j,i)) +
                                  SQR(phydro->u(IM2,k,j,i)) +
                                  SQR(phydro->u(IM3,k,j,i))) / rho;
          phydro->u(IEN,k,j,i) += (e_kin - e_kin0);
        }
      }
    }
  }
  return;
}
