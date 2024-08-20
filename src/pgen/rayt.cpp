//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file rayt.cpp
//! \brief Problem generator for point source radiation transfer in uniform medium. Works
//!        in 3D cartesian coordinates only. Used for regression test rayt/vacuum.

// C headers

// C++ headers

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"
#include "../ray_tracing/ray_tracing.hpp"

void UniformOpacity(MeshBlock *pmb, AthenaArray<Real> &chi);
int RefinementCondition(MeshBlock *pmb);
void AddRadiationSource(MeshBlock *pmb);

namespace {
  // Used for AMR only
  Real x1_src=30.0,x2_src=0.0,x3_src=0.0;
  Real chi0; // cross section per volume [length^-1]
}

//========================================================================================
//! \fn void Mesh::InitUserMeshData(ParameterInput *pin)
//! \brief Function to initialize problem-specific data in Mesh class.  Can also be used
//! to initialize variables which are global to (and therefore can be passed to) other
//! functions in this file.  Called in Mesh constructor.
//========================================================================================

void Mesh::InitUserMeshData(ParameterInput *pin) {
  chi0 = pin->GetReal("problem","chi0");
  if (adaptive) {
    EnrollUserRefinementCondition(RefinementCondition);
  }
  // Enforce timestep size
  dt = 1.0;
  return;
}

//========================================================================================
//! \fn void Mesh::UserWorkInLoop()
//! \brief Function called once every time step for user-defined work.
//========================================================================================

void Mesh::UserWorkInLoop() {
  // Enforce timestep size
  dt = 1.0;
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
  prayt->EnrollOpacityFunction(UniformOpacity);
  return;
}

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//! \brief Should be used to set initial conditions.
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        phydro->u(IDN,k,j,i) = 1.0;
        phydro->u(IM1,k,j,i) = 0.0;
        phydro->u(IM2,k,j,i) = 0.0;
        phydro->u(IM3,k,j,i) = 0.0;
      }
    }
  }
  AddRadiationSource(this);
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

void UniformOpacity(MeshBlock *pmb, AthenaArray<Real> &chi) {
  int ks=pmb->ks, ke=pmb->ke;
  int js=pmb->js, je=pmb->je;
  int is=pmb->is, ie=pmb->ie;
  for(int k=ks; k<=ke; ++k) {
    for(int j=js; j<=je; ++j) {
      for(int i=is; i<=ie; ++i) {
        for (int f=0; f<NFREQ_RAYT; f++) {
          chi(k,j,i,f) = chi0;
        }
      }
    }
  }
  return;
}

int RefinementCondition(MeshBlock *pmb) {
  Real dx = 5.0;
  int overlap = 0;
  RegionSize& bsize = pmb->block_size;
  Real x1min = x1_src - dx;
  Real x1max = x1_src + dx;
  Real x2min = x2_src - dx;
  Real x2max = x2_src + dx;
  Real x3min = x3_src - dx;
  Real x3max = x3_src + dx;

  if (((bsize.x1min < x1max) && (bsize.x1max > x1min)) &&
      ((bsize.x2min < x2max) && (bsize.x2max > x2min)) &&
      ((bsize.x3min < x3max) && (bsize.x3max > x3min))) {
    return 1;
  }

  x1min = -x1_src - dx;
  x1max = -x1_src + dx;
  x2min = -x2_src - dx;
  x2max = -x2_src + dx;
  if (((bsize.x1min < x1max) && (bsize.x1max > x1min)) &&
      ((bsize.x2min < x2max) && (bsize.x2max > x2min)) &&
      ((bsize.x3min < x3max) && (bsize.x3max > x3min))) {
    return 1;
  }

  return -1;
}


void AddRadiationSource(MeshBlock *pmb) {
  static int flag_src_added = 0;
  Mesh *pm = pmb->pmy_mesh;
  if (!pm->ray_tracing) return;
  if (flag_src_added) return;

  if (pm->adaptive) {
    pm->praytd->ResetPointSourceList();
    x1_src = 30.0*std::cos(2.0*PI*pm->time/80.0);
    x2_src = 30.0*std::sin(2.0*PI*pm->time/80.0);
    x3_src = 0.0;
    PointSourceRadiator src1(x1_src, x2_src, x3_src);
    for (int f=0; f<NFREQ_RAYT; f++) {
      src1.lum[f] = 1.0;
    }
    pm->praytd->AddPointSourceRadiator(src1);
    PointSourceRadiator src2(-x1_src, -x2_src, x3_src);
    for (int f=0; f<NFREQ_RAYT; f++) {
      src2.lum[f] = 1.0;
    }
    pm->praytd->AddPointSourceRadiator(src2);
  } else {
    PointSourceRadiator src(0.0, 0.0, 0.0);
    for (int f=0; f<NFREQ_RAYT; f++) {
      src.lum[f] = 1.0;
    }
    pm->praytd->AddPointSourceRadiator(src);
  }
  flag_src_added++;

  return;
}
