//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file hii.cpp
//! \brief Problem generator for HII region expansion. Works in 3D cartesian coordinates
//!        only. Used for regression test simple photoionization module.

// C headers

// C++ headers
#include <cmath>
#include <cstring>
#include <sstream>

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../bvals/bvals_interfaces.hpp"
#include "../eos/eos.hpp"
#include "../fft/turbulence.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../units/units.hpp"
#include "../parameter_input.hpp"
#include "../photchem/photchem.hpp"
#include "../ray_tracing/ray_tracing.hpp"
#include "../scalars/scalars.hpp"

void DiodeOutflowInnerX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                         FaceField &b, Real time, Real dt, int il, int iu, int jl, int ju,
                         int kl, int ku, int ngh);

void DiodeOutflowOuterX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                         FaceField &b, Real time, Real dt, int il, int iu, int jl, int ju,
                         int kl, int ku, int ngh);

void DiodeOutflowInnerX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                         FaceField &b, Real time, Real dt, int il, int iu, int jl, int ju,
                         int kl, int ku, int ngh);

void DiodeOutflowOuterX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                         FaceField &b, Real time, Real dt, int il, int iu, int jl, int ju,
                         int kl, int ku, int ngh);

void DiodeOutflowInnerX3(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                         FaceField &b, Real time, Real dt, int il, int iu, int jl, int ju,
                         int kl, int ku, int ngh);

void DiodeOutflowOuterX3(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                         FaceField &b, Real time, Real dt, int il, int iu, int jl, int ju,
                         int kl, int ku, int ngh);

namespace {
  constexpr Real vr_min_sh = 1.0;
  constexpr Real x_hi_min_if = 0.1;
  constexpr Real x_hi_max_if = 0.9;

  // Input parameters
  Real n_h0, n_h0_amb;  // Input density [cm^-3]
  Real rho0, rho0_amb;  // Density [code]
  Real r0;              // low-density ambient medium outside r0 [code]
  Real qi;              // Ionizing photon rate [s^-1]
  Real lumn_over_lumi;  // Luminosity ratio L_FUV/L_LyC
  Real lum[NFREQ_RAYT]; // Source luminosity [code]
  Real x1_src, x2_src, x3_src; // Source position [code]
  Real t0_src;          // Time at which source is turned on [code]
  Real t0_refine;       // Time at which refinement is turned on [code]
  Real threshold;

  // Indices for history output
  // Shell and ionization front radii, etc.
  int i_if_num, i_if_num_r, i_if_vol, i_if_vol_r;
  int i_sh_vol, i_sh_vol_r, i_sh_mass, i_sh_mass_r;
  int i_sh_mass_vr, i_sh_e_kin;

  // Mass, radial momentum, force
  int i_mass_ion, i_mass_neu;
  int i_pr_ion, i_pr_neu;

  Real HistoryIonizationFront(MeshBlock *pmb, int iout);
  Real HistoryShell(MeshBlock *pmb, int iout);
  Real HistoryMass(MeshBlock *pmb, int iout);
  Real HistoryMassEvaporationRate(MeshBlock *pmb, int iout);
  Real HistoryMomentum(MeshBlock *pmb, int iout);
  // Real HistoryForceThermal(MeshBlock *pmb, int iout);
  Real HistoryForceCentrifugal(MeshBlock *pmb, int iout);

  int RefinementCondition(MeshBlock *pmb);
  void AddRadiationSource(MeshBlock *pmb);
} // namespace

//========================================================================================
//! \fn void Mesh::InitUserMeshData(ParameterInput *pin)
//! \brief Function to initialize problem-specific data in Mesh class.  Can also be used
//! to initialize variables which are global to (and therefore can be passed to) other
//! functions in this file.  Called in Mesh constructor.
//========================================================================================

void Mesh::InitUserMeshData(ParameterInput *pin) {
  // Read in parameters related to radiation source
  n_h0 = pin->GetOrAddReal("problem", "nH0", 100.0);
  n_h0_amb = pin->GetOrAddReal("problem", "nH0_amb", 1.0);
  // Convert to code density (nh_to_rho = 1 for "ism" unit system)
  Real nh_to_rho = 1.4*Constants::hydrogen_mass_cgs/punit->code_density_cgs;
  rho0 = n_h0*nh_to_rho;
  rho0_amb = n_h0_amb*nh_to_rho;
  r0 = pin->GetReal("problem", "r0");
  x1_src = pin->GetOrAddReal("problem", "x1_src", 0.0);
  x2_src = pin->GetOrAddReal("problem", "x2_src", 0.0);
  x3_src = pin->GetOrAddReal("problem", "x3_src", 0.0);
  // Ionizing photon rates [1/s]
  qi = pin->GetReal("problem", "Qi");
  // L_non-ionizing / L_ionizing
  lumn_over_lumi = pin->GetOrAddReal("problem", "Ln_over_Li", 0.0);
  // Time at which source is turned on
  t0_src = pin->GetOrAddReal("problem", "t0_src", 0.0);

#if (NSCALARS == 0)
  std::stringstream msg;
  msg << "### FATAL ERROR in Mesh::InitUserMeshData" << std::endl
      << "This problem generator requires NSCALARS > 0."
      << std::endl;
  ATHENA_ERROR(msg);
#endif

  if (photchem) {
    if (pphotchemd->flag_update_dt_main)
      EnrollUserTimeStepFunction(&PhotochemistrySimple::PhotochemistryTimeStep);
  } else {
    std::stringstream msg;
    msg << "### FATAL ERROR in Mesh::InitUserMeshData" << std::endl
        << "This problem generator requires that Photochemsitry be turned on."
        << std::endl;
    ATHENA_ERROR(msg);
  }

  if (adaptive) {
    t0_refine = pin->GetOrAddReal("problem", "t0_refine", 0.0);
    threshold = pin->GetReal("problem", "thr");
    EnrollUserRefinementCondition(RefinementCondition);
  }

  // if (!pphotchemd->op_split) {
  //   EnrollUserExplicitSourceFunction(&Photochemistry::UpdateSourceTerms);
  // }

  // Set diode outflow boundaries
  EnrollUserBoundaryFunction(BoundaryFace::inner_x1, DiodeOutflowInnerX1);
  EnrollUserBoundaryFunction(BoundaryFace::outer_x1, DiodeOutflowOuterX1);
  EnrollUserBoundaryFunction(BoundaryFace::inner_x2, DiodeOutflowInnerX2);
  EnrollUserBoundaryFunction(BoundaryFace::outer_x2, DiodeOutflowOuterX2);
  EnrollUserBoundaryFunction(BoundaryFace::inner_x3, DiodeOutflowInnerX3);
  EnrollUserBoundaryFunction(BoundaryFace::outer_x3, DiodeOutflowOuterX3);

  // Enroll new history variables
  int n_user_hst = 15, i_user_hst = 0;

  AllocateUserHistoryOutput(n_user_hst);

  // Ionization front radius
  i_if_num = i_user_hst;
  EnrollUserHistoryOutput(i_user_hst++, HistoryIonizationFront, "IF_num");
  i_if_num_r = i_user_hst;
  EnrollUserHistoryOutput(i_user_hst++, HistoryIonizationFront, "IF_num_r");
  i_if_vol = i_user_hst;
  EnrollUserHistoryOutput(i_user_hst++, HistoryIonizationFront, "IF_vol");
  i_if_vol_r = i_user_hst;
  EnrollUserHistoryOutput(i_user_hst++, HistoryIonizationFront, "IF_vol_r");

  // Shell
  i_sh_vol = i_user_hst;
  EnrollUserHistoryOutput(i_user_hst++, HistoryShell, "sh_vol");
  i_sh_vol_r = i_user_hst;
  EnrollUserHistoryOutput(i_user_hst++, HistoryShell, "sh_vol_r");
  i_sh_mass = i_user_hst;
  EnrollUserHistoryOutput(i_user_hst++, HistoryShell, "sh_mass");
  i_sh_mass_r = i_user_hst;
  EnrollUserHistoryOutput(i_user_hst++, HistoryShell, "sh_mass_r");
  i_sh_mass_vr = i_user_hst;
  EnrollUserHistoryOutput(i_user_hst++, HistoryShell, "sh_mass_vr");
  i_sh_e_kin = i_user_hst;
  EnrollUserHistoryOutput(i_user_hst++, HistoryShell, "sh_e_kin");

  // Mass
  i_mass_neu = i_user_hst;
  EnrollUserHistoryOutput(i_user_hst++, HistoryMass, "mass_neu");
  i_mass_ion = i_user_hst;
  EnrollUserHistoryOutput(i_user_hst++, HistoryMass, "mass_ion");
  EnrollUserHistoryOutput(i_user_hst++, HistoryMassEvaporationRate, "dot_mass_ion");

  // Radial momentum
  i_pr_neu = i_user_hst;
  EnrollUserHistoryOutput(i_user_hst++, HistoryMomentum, "pr_neu");
  i_pr_ion = i_user_hst;
  EnrollUserHistoryOutput(i_user_hst++, HistoryMomentum, "pr_ion");

  return;
}


//========================================================================================
//! \fn void Mesh::UserWorkInLoop()
//! \brief Function called once every time step for user-defined work.
//========================================================================================

void Mesh::UserWorkInLoop() {
  if (time >= t0_src) AddRadiationSource(my_blocks(0));
  return;
}

//========================================================================================
//! \fn void Mesh::UserWorkAfterLoop(ParameterInput *pin)
//! \brief Function called after main loop is finished for user-defined work.
//========================================================================================

void __attribute__((weak)) Mesh::UserWorkAfterLoop(ParameterInput *pin) {
  // do nothing
  return;
}

// 4x members of MeshBlock class:

//========================================================================================
//! \fn void MeshBlock::InitUserMeshBlockData(ParameterInput *pin)
//! \brief Function to initialize problem-specific data in MeshBlock class.  Can also be
//! used to initialize variables which are global to other functions in this file.
//! Called in MeshBlock constructor before ProblemGenerator.
//========================================================================================

void MeshBlock::InitUserMeshBlockData(ParameterInput *pin) {
  if (pmy_mesh->pphotchemd->mode == PhotochemistryMode::simple) {
    PhotochemistrySimple *ppc = static_cast<PhotochemistrySimple*>(pphotchem);
    if (ppc->bookkeeping) {
      // Set output variables
      int num_user_variables = 2;
      AllocateUserOutputVariables(num_user_variables);
      ppc->rho_hi_dot.InitWithShallowSlice(user_out_var,4,0,1);
      SetUserOutputVariableName(0, "rho_hi_dot");
      ppc->x_hi_before.InitWithShallowSlice(user_out_var,4,1,1);
      SetUserOutputVariableName(1, "x_hi_before");
    }
  }

  return;
}

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//! \brief Should be used to set initial conditions.
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  if (std::strcmp(COORDINATE_SYSTEM, "cartesian") != 0) {
    std::stringstream msg;
    msg << "### FATAL ERROR in hii.cpp ProblemGenerator" << std::endl
        << "HII region test only compatible with cartesian coord" << std::endl;
    ATHENA_ERROR(msg);
  }

  // TODO(JGKIM): This is not needed and will omitted in the future.
  if (std::strcmp(pmy_mesh->punit->unit_system.c_str(), "ism") != 0) {
    std::stringstream msg;
    msg << "### FATAL ERROR in hii.cpp ProblemGenerator" << std::endl
        << "This problem is compatible only with ism unit system" << std::endl;
    ATHENA_ERROR(msg);
  }

  if (pmy_mesh->pphotchemd->mode == PhotochemistryMode::simple) {
    PhotochemistrySimple *ppc = static_cast<PhotochemistrySimple*>(pphotchem);
    if (!ppc->bookkeeping) {
      std::stringstream msg;
      msg << "### FATAL ERROR in hii.cpp ProblemGenerator" << std::endl
          << "This problem requires bookkeeping (simple photochemistry) turned on."
          << std::endl;
      ATHENA_ERROR(msg);
    }
  }

  Real press;
  Real igm1 = 1.0/(peos->GetGamma() - 1.0);
  if (pmy_mesh->pphotchemd->mode == PhotochemistryMode::simple) {
    PhotochemistrySimple *ppc = static_cast<PhotochemistrySimple*>(pphotchem);
    Real pressure_cgs_inv = 1.0/pmy_mesh->punit->code_pressure_cgs;
    // Fully neutral HI gas
    Real x_hi0 = 1.0;
    Real rho_hi0 = rho0*x_hi0;
    Real rho_hi0_amb = rho0_amb*x_hi0;
    Real tgas0 = ppc->GetTemperature(x_hi0);
    for (int k=ks; k<=ke; k++) {
      Real x3 = pcoord->x3v(k);
      for (int j=js; j<=je; j++) {
        Real x2 = pcoord->x2v(j);
        for (int i=is; i<=ie; i++) {
          Real x1= pcoord->x1v(i);
          Real r = std::sqrt(SQR(x1 - x1_src) + SQR(x2 - x1_src) + SQR(x3 - x3_src));
          if (r <= r0) {
            phydro->u(IDN,k,j,i) = rho0;
            press = ppc->GetPressureCGS(n_h0, tgas0, x_hi0)*pressure_cgs_inv;
            phydro->u(IEN,k,j,i) = press*igm1;
            pscalars->s(IHI,k,j,i) = rho_hi0;
          } else {
            phydro->u(IDN,k,j,i) = rho0_amb;
            press = ppc->GetPressureCGS(n_h0_amb, tgas0, x_hi0)*pressure_cgs_inv;
            phydro->u(IEN,k,j,i) = press*igm1;
            pscalars->s(IHI,k,j,i) = rho_hi0_amb;
          }
          phydro->u(IM1,k,j,i) = 0.0;
          phydro->u(IM2,k,j,i) = 0.0;
          phydro->u(IM3,k,j,i) = 0.0;
        }
      }
    }
  }
  if (pmy_mesh->time >= t0_src) AddRadiationSource(this);
  return;
}

//========================================================================================
//! \fn void MeshBlock::UserWorkInLoop()
//! \brief Function called once every time step for user-defined work.
//========================================================================================

void __attribute__((weak)) MeshBlock::UserWorkInLoop() {
  // do nothing
  return;
}

//========================================================================================
//! \fn void MeshBlock::UserWorkBeforeOutput(ParameterInput *pin)
//! \brief Function called before generating output files
//========================================================================================

void __attribute__((weak)) MeshBlock::UserWorkBeforeOutput(ParameterInput *pin) {
  // do nothing
  return;
}

void DiodeOutflowInnerX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                         FaceField &b, Real time, Real dt, int il, int iu, int jl, int ju,
                         int kl, int ku, int ngh) {
  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
#pragma omp simd
      for (int i=1; i<=ngh; ++i) {
        prim(IDN,k,j,(il-i)) = prim(IDN,k,j,il);
        prim(IVX,k,j,(il-i)) = std::min(prim(IVX,k,j,il), static_cast<Real>(0.0));
        prim(IVY,k,j,(il-i)) = prim(IVY,k,j,il);
        prim(IVZ,k,j,(il-i)) = prim(IVZ,k,j,il);
        if (NON_BAROTROPIC_EOS)
          prim(IPR,k,j,(il-i)) = prim(IPR,k,j,il);
      }
    }
  }
  if (NSCALARS > 0) {
    for (int n=0; n<NSCALARS; ++n) {
      for (int k=kl; k<=ku; ++k) {
        for (int j=jl; j<=ju; ++j) {
#pragma omp simd
          for (int i=1; i<=ngh; ++i) {
            pmb->pscalars->r(n,k,j,(il-i)) = pmb->pscalars->r(n,k,j,il);
          }
        }
      }
    }
  }
  if (!MAGNETIC_FIELDS_ENABLED) {
    return;
  }

  // copy face-centered magnetic fields into ghost zones
  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
#pragma omp simd
      for (int i=1; i<=ngh; ++i) {
        b.x1f(k,j,(il-i)) = b.x1f(k,j,il);
      }
    }
  }
  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju+1; ++j) {
#pragma omp simd
      for (int i=1; i<=ngh; ++i) {
        b.x2f(k,j,(il-i)) = b.x2f(k,j,il);
      }
    }
  }
  for (int k=kl; k<=ku+1; ++k) {
    for (int j=jl; j<=ju; ++j) {
#pragma omp simd
      for (int i=1; i<=ngh; ++i) {
        b.x3f(k,j,(il-i)) = b.x3f(k,j,il);
      }
    }
  }
  return;
}

void DiodeOutflowOuterX1(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                         FaceField &b, Real time, Real dt, int il, int iu, int jl, int ju,
                         int kl, int ku, int ngh) {
  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
#pragma omp simd
      for (int i=1; i<=ngh; ++i) {
        prim(IDN,k,j,(iu+i)) = prim(IDN,k,j,iu);
        prim(IVX,k,j,(iu+i)) = std::max(prim(IVX,k,j,iu), static_cast<Real>(0.0));
        prim(IVY,k,j,(iu+i)) = prim(IVY,k,j,iu);
        prim(IVZ,k,j,(iu+i)) = prim(IVZ,k,j,iu);
        if (NON_BAROTROPIC_EOS)
          prim(IPR,k,j,(iu+i)) = prim(IPR,k,j,iu);
      }
    }
  }
  if (NSCALARS > 0) {
    for (int n=0; n<NSCALARS; ++n) {
      for (int k=kl; k<=ku; ++k) {
        for (int j=jl; j<=ju; ++j) {
#pragma omp simd
          for (int i=1; i<=ngh; ++i) {
            pmb->pscalars->r(n,k,j,(iu+i)) = pmb->pscalars->r(n,k,j,iu);
          }
        }
      }
    }
  }
  if (!MAGNETIC_FIELDS_ENABLED) {
    return;
  }

  // copy face-centered magnetic fields into ghost zones
  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju; ++j) {
#pragma omp simd
      for (int i=1; i<=ngh; ++i) {
        b.x1f(k,j,(iu+i+1)) = b.x1f(k,j,(iu+1));
      }
    }
  }
  for (int k=kl; k<=ku; ++k) {
    for (int j=jl; j<=ju+1; ++j) {
#pragma omp simd
      for (int i=1; i<=ngh; ++i) {
        b.x2f(k,j,(iu+i)) = b.x2f(k,j,iu);
      }
    }
  }
  for (int k=kl; k<=ku+1; ++k) {
    for (int j=jl; j<=ju; ++j) {
#pragma omp simd
      for (int i=1; i<=ngh; ++i) {
        b.x3f(k,j,(iu+i)) = b.x3f(k,j,iu);
      }
    }
  }
  return;
}

void DiodeOutflowInnerX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                         FaceField &b, Real time, Real dt, int il, int iu, int jl, int ju,
                         int kl, int ku, int ngh) {
  for (int k=kl; k<=ku; ++k) {
    for (int j=1; j<=ngh; ++j) {
#pragma omp simd
     for (int i=il; i<=iu; ++i) {
       prim(IDN,k,(jl-j),i) = prim(IDN,k,jl,i);
       prim(IVX,k,(jl-j),i) = prim(IVX,k,jl,i);
       prim(IVY,k,(jl-j),i) = std::min(prim(IVY,k,jl,i), static_cast<Real>(0.0));
       prim(IVZ,k,(jl-j),i) = prim(IVZ,k,jl,i);
       if (NON_BAROTROPIC_EOS)
         prim(IPR,k,(jl-j),i) = prim(IPR,k,jl,i);
      }
    }
  }
  if (NSCALARS > 0) {
    for (int n=0; n<NSCALARS; ++n) {
      for (int k=kl; k<=ku; ++k) {
        for (int j=1; j<=ngh; ++j) {
#pragma omp simd
          for (int i=il; i<=iu; ++i) {
            pmb->pscalars->r(n,k,(jl-j),i) = pmb->pscalars->r(n,k,jl,i);
          }
        }
      }
    }
  }
  if (!MAGNETIC_FIELDS_ENABLED) {
    return;
  }

  // copy face-centered magnetic fields into ghost zones
  for (int k=kl; k<=ku; ++k) {
    for (int j=1; j<=ngh; ++j) {
#pragma omp simd
      for (int i=il; i<=iu+1; ++i) {
        b.x1f(k,(jl-j),i) = b.x1f(k,jl,i);
      }
    }
  }
  for (int k=kl; k<=ku; ++k) {
    for (int j=1; j<=ngh; ++j) {
#pragma omp simd
      for (int i=il; i<=iu; ++i) {
        b.x2f(k,(jl-j),i) = b.x2f(k,jl,i);
      }
    }
  }
  for (int k=kl; k<=ku+1; ++k) {
    for (int j=1; j<=ngh; ++j) {
#pragma omp simd
      for (int i=il; i<=iu; ++i) {
        b.x3f(k,(jl-j),i) = b.x3f(k,jl,i);
      }
    }
  }
  return;
}

void DiodeOutflowOuterX2(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                         FaceField &b, Real time, Real dt, int il, int iu, int jl, int ju,
                         int kl, int ku, int ngh) {
  for (int k=kl; k<=ku; ++k) {
    for (int j=1; j<=ngh; ++j) {
#pragma omp simd
      for (int i=il; i<=iu; ++i) {
        prim(IDN,k,(ju+j),i) = prim(IDN,k,ju,i);
        prim(IVX,k,(ju+j),i) = prim(IVX,k,ju,i);
        prim(IVY,k,(ju+j),i) = std::max(prim(IVY,k,ju,i), static_cast<Real>(0.0));
        prim(IVZ,k,(ju+j),i) = prim(IVZ,k,ju,i);
        if (NON_BAROTROPIC_EOS)
          prim(IPR,k,(ju+j),i) = prim(IPR,k,ju,i);
      }
    }
  }
  if (NSCALARS > 0) {
    for (int n=0; n<NSCALARS; ++n) {
      for (int k=kl; k<=ku; ++k) {
        for (int j=1; j<=ngh; ++j) {
#pragma omp simd
          for (int i=il; i<=iu; ++i) {
            pmb->pscalars->r(n,k,(ju+j),i) = pmb->pscalars->r(n,k,ju,i);
          }
        }
      }
    }
  }
  if (!MAGNETIC_FIELDS_ENABLED) {
    return;
  }

  // copy face-centered magnetic fields into ghost zones
  for (int k=kl; k<=ku; ++k) {
    for (int j=1; j<=ngh; ++j) {
#pragma omp simd
      for (int i=il; i<=iu+1; ++i) {
        b.x1f(k,(ju+j  ),i) = b.x1f(k,(ju  ),i);
      }
    }
  }
  for (int k=kl; k<=ku; ++k) {
    for (int j=1; j<=ngh; ++j) {
#pragma omp simd
      for (int i=il; i<=iu; ++i) {
        b.x2f(k,(ju+j+1),i) = b.x2f(k,(ju+1),i);
      }
    }
  }
  for (int k=kl; k<=ku+1; ++k) {
    for (int j=1; j<=ngh; ++j) {
#pragma omp simd
      for (int i=il; i<=iu; ++i) {
        b.x3f(k,(ju+j  ),i) = b.x3f(k,(ju  ),i);
      }
    }
  }
  return;
}

void DiodeOutflowInnerX3(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                         FaceField &b, Real time, Real dt, int il, int iu, int jl, int ju,
                         int kl, int ku, int ngh) {
  for (int k=1; k<=ngh; ++k) {
    for (int j=jl; j<=ju; ++j) {
#pragma omp simd
      for (int i=il; i<=iu; ++i) {
        prim(IDN,(kl-k),j,i) = prim(IDN,kl,j,i);
        prim(IVX,(kl-k),j,i) = prim(IVX,kl,j,i);
        prim(IVY,(kl-k),j,i) = prim(IVY,kl,j,i);
        prim(IVZ,(kl-k),j,i) = std::min(prim(IVZ,kl,j,i), static_cast<Real>(0.0));
        if (NON_BAROTROPIC_EOS)
          prim(IPR,(kl-k),j,i) = prim(IPR,kl,j,i);
      }
    }
  }
  if (NSCALARS > 0) {
    for (int n=0; n<NSCALARS; ++n) {
      for (int k=1; k<=ngh; ++k) {
        for (int j=jl; j<=ju; ++j) {
#pragma omp simd
          for (int i=il; i<=iu; ++i) {
            pmb->pscalars->r(n,(kl-k),j,i) = pmb->pscalars->r(n,kl,j,i);
          }
        }
      }
    }
  }
  if (!MAGNETIC_FIELDS_ENABLED) {
    return;
  }

  // copy face-centered magnetic fields into ghost zones
  for (int k=1; k<=ngh; ++k) {
    for (int j=jl; j<=ju; ++j) {
#pragma omp simd
      for (int i=il; i<=iu+1; ++i) {
        b.x1f((kl-k),j,i) = b.x1f(kl,j,i);
      }
    }
  }
  for (int k=1; k<=ngh; ++k) {
    for (int j=jl; j<=ju+1; ++j) {
#pragma omp simd
      for (int i=il; i<=iu; ++i) {
        b.x2f((kl-k),j,i) = b.x2f(kl,j,i);
      }
    }
  }
  for (int k=1; k<=ngh; ++k) {
    for (int j=jl; j<=ju; ++j) {
#pragma omp simd
      for (int i=il; i<=iu; ++i) {
        b.x3f((kl-k),j,i) = b.x3f(kl,j,i);
      }
    }
  }
  return;
}

void DiodeOutflowOuterX3(MeshBlock *pmb, Coordinates *pco, AthenaArray<Real> &prim,
                         FaceField &b, Real time, Real dt, int il, int iu, int jl, int ju,
                         int kl, int ku, int ngh) {
  for (int k=1; k<=ngh; ++k) {
    for (int j=jl; j<=ju; ++j) {
#pragma omp simd
      for (int i=il; i<=iu; ++i) {
        prim(IDN,(ku+k),j,i) = prim(IDN,ku,j,i);
        prim(IVX,(ku+k),j,i) = prim(IVX,ku,j,i);
        prim(IVY,(ku+k),j,i) = prim(IVY,ku,j,i);
        prim(IVZ,(ku+k),j,i) = std::max(prim(IVZ,ku,j,i), static_cast<Real>(0.0));
        if (NON_BAROTROPIC_EOS)
          prim(IPR,(ku+k),j,i) = prim(IPR,ku,j,i);
      }
    }
  }
  if (NSCALARS > 0) {
    for (int n=0; n<NSCALARS; ++n) {
      for (int k=1; k<=ngh; ++k) {
        for (int j=jl; j<=ju; ++j) {
#pragma omp simd
          for (int i=il; i<=iu; ++i) {
            pmb->pscalars->r(n,(ku+k),j,i) = pmb->pscalars->r(n,ku,j,i);
          }
        }
      }
    }
  }
  if (!MAGNETIC_FIELDS_ENABLED) {
    return;
  }

  // copy face-centered magnetic fields into ghost zones
  for (int k=1; k<=ngh; ++k) {
    for (int j=jl; j<=ju; ++j) {
#pragma omp simd
      for (int i=il; i<=iu+1; ++i) {
        b.x1f((ku+k  ),j,i) = b.x1f((ku  ),j,i);
      }
    }
  }
  for (int k=1; k<=ngh; ++k) {
    for (int j=jl; j<=ju+1; ++j) {
#pragma omp simd
      for (int i=il; i<=iu; ++i) {
        b.x2f((ku+k  ),j,i) = b.x2f((ku  ),j,i);
      }
    }
  }
  for (int k=1; k<=ngh; ++k) {
    for (int j=jl; j<=ju; ++j) {
#pragma omp simd
      for (int i=il; i<=iu; ++i) {
        b.x3f((ku+k+1),j,i) = b.x3f((ku+1),j,i);
      }
    }
  }
  return;
}

namespace {
void AddRadiationSource(MeshBlock *pmb) {
  static int flag_src_added = 0;
  Mesh *pm = pmb->pmy_mesh;
  if (!pm->ray_tracing) return;
  if (flag_src_added) return;

  // TODO(JGKIM): make this to work for more general cases
  PointSourceRadiator src(x1_src, x2_src, x3_src);
  if (pm->pphotchemd->mode == PhotochemistryMode::simple) {
    PhotochemistrySimple *ppc = static_cast<PhotochemistrySimple*>
      (pm->my_blocks(0)->pphotchem);
    lum[ILyC] = (qi/pm->punit->second_code)*ppc->GetMeanPhotonEnergy(ILyC);
    if (NFREQ_RAYT > 1) {
      lum[IFUV] = lum[ILyC]*lumn_over_lumi;
    }
  }

  for (int f=0; f<NFREQ_RAYT; ++f) {
    src.lum[f] = lum[f];
  }
  pm->praytd->AddPointSourceRadiator(src);
  flag_src_added++;

  return;
}


//========================================================================================
//! \fn Real HistoryIonizationFront(MeshBlock *pmb, int iout)
//! \brief History variables for cells with intermediate ionization fraction.
//========================================================================================

Real HistoryIonizationFront(MeshBlock *pmb, int iout) {
  int is=pmb->is, ie=pmb->ie;
  int js=pmb->js, je=pmb->je;
  int ks=pmb->ks, ke=pmb->ke;
  Real dvol = pmb->pcoord->GetCellVolume(ks,js,is);
  AthenaArray<Real> rho, rho_hi;
  rho.InitWithShallowSlice(pmb->phydro->u,4,IDN,1);
  rho_hi.InitWithShallowSlice(pmb->pscalars->s,4,IHI,1);

  Real sum = 0.0;
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie; ++i) {
        Real x_hi = rho_hi(k,j,i) / rho(k,j,i);
        if ((x_hi >= x_hi_min_if) && (x_hi < x_hi_max_if)) {
          Real x1 = pmb->pcoord->x1v(i);
          Real x2 = pmb->pcoord->x2v(j);
          Real x3 = pmb->pcoord->x3v(k);
          Real r = std::sqrt(SQR(x1 - x1_src) + SQR(x2 - x2_src) + SQR(x3 - x3_src));
          if (iout == i_if_num) {   // simple arithmetic average radius
            sum += 1.0;
          } else if (iout == i_if_num_r) {
            sum += r;
          } else if (iout == i_if_vol) {  // volume-weighted average radius
            sum += dvol;
          } else if (iout == i_if_vol_r) {
            sum += r*dvol;
          }
        }
      }
    }
  }
  return sum;
}

//========================================================================================
//! \fn Real HistoryShell(MeshBlock *pmb, int iout)
//! \brief History variables for shell
//========================================================================================

Real HistoryShell(MeshBlock *pmb, int iout) {
  int is=pmb->is, ie=pmb->ie;
  int js=pmb->js, je=pmb->je;
  int ks=pmb->ks, ke=pmb->ke;
  Real dvol = pmb->pcoord->GetCellVolume(ks,js,is);
  AthenaArray<Real> rho, rho_hi;
  rho.InitWithShallowSlice(pmb->phydro->u,4,IDN,1);
  rho_hi.InitWithShallowSlice(pmb->pscalars->s,4,IHI,1);

  Real sum = 0.0;
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie; ++i) {
        Real rho_ = rho(k,j,i);
        Real rho_hi_ = rho_hi(k,j,i);
        Real x_hi = rho_hi_ / rho_;
        Real x1 = pmb->pcoord->x1v(i);
        Real x2 = pmb->pcoord->x2v(j);
        Real x3 = pmb->pcoord->x3v(k);
        Real v1 = pmb->phydro->w(IVX,k,j,i);
        Real v2 = pmb->phydro->w(IVY,k,j,i);
        Real v3 = pmb->phydro->w(IVZ,k,j,i);
        Real r = std::sqrt(SQR(x1-x1_src) + SQR(x2-x2_src) + SQR(x3-x3_src));
        Real vr = (v1*(x1-x1_src) + v2*(x2-x2_src) + v3*(x3-x3_src)) / r;
        if ((x_hi >= x_hi_max_if) && (vr > vr_min_sh)) {
          if (iout == i_sh_vol) {
            sum += dvol;
          } else if (iout == i_sh_vol_r) {
            sum += dvol*r;
          } else if (iout == i_sh_mass) {
            sum += dvol*rho_;
          } else if (iout == i_sh_mass_r) {
            sum += dvol*rho_*r;
          } else if (iout == i_sh_mass_vr) {
            sum += dvol*rho_*vr;
          } else if (iout == i_sh_e_kin) {
            Real e_kin = 0.5*rho_*(SQR(v1) + SQR(v2) + SQR(v3));
            sum += dvol*e_kin;
          }
        }
      }
    }
  }
  return sum;
}

//========================================================================================
//! \fn Real HistoryMass(MeshBlock *pmb, int iout)
//! \brief History variables for gas mass.
//========================================================================================

Real HistoryMass(MeshBlock *pmb, int iout) {
  int is=pmb->is, ie=pmb->ie;
  int js=pmb->js, je=pmb->je;
  int ks=pmb->ks, ke=pmb->ke;
  Real dvol = pmb->pcoord->GetCellVolume(ks,js,is);
  AthenaArray<Real> rho, rho_hi;
  rho.InitWithShallowSlice(pmb->phydro->u,4,IDN,1);
  rho_hi.InitWithShallowSlice(pmb->pscalars->s,4,IHI,1);

  Real sum = 0.0;
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie; ++i) {
        Real x_hi = rho_hi(k,j,i) / rho(k,j,i);
        if (((iout == i_mass_neu) && (x_hi > 0.5)) ||
            ((iout == i_mass_ion) && (x_hi <= 0.5))) {
          sum += dvol*rho(k,j,i);
        }
      }
    }
  }
  return sum;
}

//========================================================================================
//! \fn Real HistoryMassEvaporationRate(MeshBlock *pmb, int iout)
//! \brief History variables for mass evaporation rate.
//========================================================================================

Real HistoryMassEvaporationRate(MeshBlock *pmb, int iout) {
  int is=pmb->is, ie=pmb->ie;
  int js=pmb->js, je=pmb->je;
  int ks=pmb->ks, ke=pmb->ke;
  Real dvol = pmb->pcoord->GetCellVolume(ks,js,is);

  PhotochemistrySimple *ppc = static_cast<PhotochemistrySimple*>(pmb->pphotchem);
  Real sum = 0.0;
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie; ++i) {
        sum += dvol*ppc->rho_hi_dot(k,j,i);
      }
    }
  }
  return -sum;
}

//========================================================================================
//! \fn Real HistoryMass(MeshBlock *pmb, int iout)
//! \brief History variables for radial momentum.
//========================================================================================

Real HistoryMomentum(MeshBlock *pmb, int iout) {
  int is=pmb->is, ie=pmb->ie;
  int js=pmb->js, je=pmb->je;
  int ks=pmb->ks, ke=pmb->ke;
  Real dvol = pmb->pcoord->GetCellVolume(ks,js,is);
  AthenaArray<Real> rho, rho_hi;
  rho.InitWithShallowSlice(pmb->phydro->u,4,IDN,1);
  rho_hi.InitWithShallowSlice(pmb->pscalars->s,4,IHI,1);

  Real sum = 0.0;
  for (int k=ks; k<=ke; ++k) {
    for (int j=js; j<=je; ++j) {
      for (int i=is; i<=ie; ++i) {
        Real x_hi = rho_hi(k,j,i)/rho(k,j,i);
        if (((iout == i_pr_neu) && (x_hi > 0.5)) ||
            ((iout == i_pr_ion) && (x_hi <= 0.5))) {
          Real rhov1 = pmb->phydro->u(IM1,k,j,i);
          Real rhov2 = pmb->phydro->u(IM2,k,j,i);
          Real rhov3 = pmb->phydro->u(IM3,k,j,i);
          Real x1 = pmb->pcoord->x1v(i);
          Real x2 = pmb->pcoord->x2v(j);
          Real x3 = pmb->pcoord->x3v(k);
          Real r = std::sqrt(SQR(x1 - x1_src) + SQR(x2 - x2_src) + SQR(x3 - x3_src));
          Real rhovr = (rhov1*(x1-x1_src) + rhov2*(x2-x2_src) + rhov3*(x3-x3_src))/r;
          sum += dvol*rhovr;
        }
      }
    }
  }
  return sum;
}


int RefinementCondition(MeshBlock *pmb) {
  AthenaArray<Real> &w = pmb->phydro->w;
  // AthenaArray<Real> &r = pmb->pscalars->r;
  Real maxeps = 0.0;
  for (int k=pmb->ks-1; k<=pmb->ke+1; k++) {
    for (int j=pmb->js-1; j<=pmb->je+1; j++) {
      for (int i=pmb->is-1; i<=pmb->ie+1; i++) {
        Real eps = std::sqrt(SQR(0.5*(w(IPR,k,j,i+1) - w(IPR,k,j,i-1)))
                             +SQR(0.5*(w(IPR,k,j+1,i) - w(IPR,k,j-1,i)))
                             +SQR(0.5*(w(IPR,k+1,j,i) - w(IPR,k-1,j,i))))/w(IPR,k,j,i);
        // Real eps = std::sqrt(SQR(0.5*(r(IHI,k,j,i+1) - r(IHI,k,j,i-1))) +
        //                      SQR(0.5*(r(IHI,k,j+1,i) - r(IHI,k,j-1,i))) +
        //                      SQR(0.5*(r(IHI,k+1,j,i) - r(IHI,k-1,j,i))));
        maxeps = std::max(maxeps, eps);
      }
    }
  }

  if (maxeps > threshold) return 1;
  if (maxeps < 0.25*threshold) return -1;
  return 0;

  // Real dx = 5.0;
  // RegionSize& bsize = pmb->block_size;
  // Real x1min = x1_src - dx;
  // Real x1max = x1_src + dx;
  // Real x2min = x2_src - dx;
  // Real x2max = x2_src + dx;
  // Real x3min = x3_src - dx;
  // Real x3max = x3_src + dx;
  // Real time = pmb->pmy_mesh->time;

  // if (((bsize.x1min < x1max) && (bsize.x1max > x1min)) &&
  //     ((bsize.x2min < x2max) && (bsize.x2max > x2min)) &&
  //     ((bsize.x3min < x3max) && (bsize.x3max > x3min)) &&
  //     (time > t0_refine)) {
  //   return 1;
  // } else {
  //   return 0;
  // }
  //  return -1;
}

} // namespace
