//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file photchem.cpp
//! \brief Implementation of functions in class Photochemistry and PhotochemistryDriver
//========================================================================================

// C headers

// C++ headers
#include <cmath>
#include <string>

// Athena++ headers
#include "../athena.hpp"
#include "../bvals/bvals.hpp"
#include "../defs.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../units/units.hpp"
#include "../parameter_input.hpp"
#include "../scalars/scalars.hpp"
#include "photchem.hpp"

Photochemistry::Photochemistry(MeshBlock *pmy_block, ParameterInput *pin) :
  pmb(pmy_block),
  punit(pmb->pmy_mesh->punit),
  mode_(pmb->pmy_mesh->pphotchemd->mode) {
  SetRadiationMoments();
}

void Photochemistry::SetRadiationMoments() {
  // TODO(JGKIM): Make sure that the array shape is consistent
  // What about nfreq?
  if (pmb->pmy_mesh->ray_tracing) {
    rad_mom_.InitWithShallowSlice(pmb->prayt->rad_mom,5,0,4);
  } else {
    // TODO(JGKIM): Check why this is here..? Perhaps we don't need rad_mom_ at all..
    // unless photochemistry is coulped with other radiation modules
    rad_mom_.NewAthenaArray(4, pmb->ncells3, pmb->ncells2, pmb->ncells1, NFREQ);
  }
}

PhotochemistryDriver::PhotochemistryDriver(Mesh *pm, ParameterInput *pin) :
  flag_op_split(pin->GetOrAddBoolean("photchem", "flag_op_split", true)),
  flag_rad_force(pin->GetOrAddBoolean("photchem","flag_rad_force",true)),
  flag_update_dt_main(pin->GetOrAddBoolean("photchem","flag_update_dt_main",true)),
  pmy_mesh_(pm) {
  SetMode(pin);
  punit = pm->punit;
}

PhotochemistryDriver::~PhotochemistryDriver() {
}

void PhotochemistryDriver::SetMode(ParameterInput *pin) {
  std::string str_mode = pin->GetString("photchem", "mode");
  if (str_mode == "simple") {
    mode = PhotochemistryMode::simple;
  } else { // unsupported mode
    std::stringstream msg;
    msg << "### FATAL ERROR in PhotochemistryDriver" << std::endl
        << "Unrecognized mode = '" << str_mode
        << "' in <photchem> input block '" << std::endl;
    ATHENA_ERROR(msg);
  }
  return;
}

Photochemistry* PhotochemistryDriver::CreatePhotochemistry(MeshBlock *pmb,
                                                           ParameterInput *pin) {
  Photochemistry *pphotchem=nullptr;
  if (mode == PhotochemistryMode::simple) {
    pphotchem = new PhotochemistrySimple(pmb, pin);
  }
  return pphotchem;
}

void PhotochemistryDriver::UpdateRatesAndTimeStep() {
  return;
}
