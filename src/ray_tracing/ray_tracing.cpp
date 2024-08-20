//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file ray_tracing.cpp
//! \brief Implementation of functions in classes RayTracing and RayTracingDriver
//========================================================================================

// C/C++ Standard Libraries
#include <algorithm> // min, max
#include <cmath>
#include <iomanip> // setprecision
#include <iostream>
#include <sstream>
#include <string>  // string

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../bvals/bvals.hpp"
#include "../coordinates/coordinates.hpp"
#include "../mesh/mesh.hpp"
#include "../units/units.hpp"
#include "../parameter_input.hpp"
#include "ray_tracing.hpp"

// MPI header
#ifdef MPI_PARALLEL
#include <mpi.h>
#endif

namespace {
  //! Number of rays (or pixels) at HEALPix level 1
  constexpr int nray_base = 12;

  // N_pixel(level) = 12 x 4^level
  // N_pixel(30) ~ 1.4e19 while uint64_max = 0xFFFFFFFFFFFFFFFF ~ 1.8e19
  // Therefore, maximum HEALPix level cannot be larger than 30 for a single source.
  // Note also that destroy_counter_max = N_src * N_pixel and
  // there can be large number of sources.
  constexpr int MAX_HEALPIX_LEVEL = 25;

  // This number is used to set the minimum (absolute) value of ray direction vector,
  // e.g., n_1 = max(abs(n1), small_number), and hence avoid division by zero. It is small
  // enough that edge cases don't occur frequently, but also large enough that we have
  // some precision left to do additions like x0 = x1_src + dist*n1, etc.
  static constexpr Real small_number = 1e-10;

  static constexpr Real b5_inv = 0.3333333333333333;
} // namespace

//--------------------------------------------------------------------------------------
//! \fn RayTracing::RayTracing(MeshBlock *pmb, ParameterInput *pin)
//! \brief Constructs a RayTracing instance.

RayTracing::RayTracing(MeshBlock *pmy_block, ParameterInput *pin) :
  pmb(pmy_block),
  tau_max_(pin->GetOrAddReal("ray_tracing", "tau_max", tau_max_def_)) {
  // Initialize arrays
  rad_mom.NewAthenaArray(4, pmb->ncells3, pmb->ncells2, pmb->ncells1, NFREQ);
  chi_rayt.NewAthenaArray(pmb->ncells3, pmb->ncells2, pmb->ncells1, NFREQ);
  rad_mom0_fshld.NewAthenaArray(1, pmb->ncells3, pmb->ncells2, pmb->ncells1);
  den_rayt.NewAthenaArray(pmb->ncells3, pmb->ncells2, pmb->ncells1, 1);
  flag_opacity_enrolled_ = false;
  flag_density_enrolled_ = false;
  // Cell volume
  dvol_ = pmb->pcoord->GetCellVolume(pmb->ks, pmb->js, pmb->is);
}


//--------------------------------------------------------------------------------------
//! \fn RayTracing::~RayTracing()
//! \brief Destroys a RayTracing instance.

RayTracing::~RayTracing() {
  rad_mom.DeleteAthenaArray();
  chi_rayt.DeleteAthenaArray();
  rad_mom0_fshld.DeleteAthenaArray();
  den_rayt.DeleteAthenaArray();
}


//--------------------------------------------------------------------------------------
//! \fn RayTracing::EnrollOpacityFunction()
//! \brief Enrolls a function to calculate opacity.

void RayTracing::EnrollOpacityFunction(RayTracingOpacityFunc my_func) {
  UpdateOpacity_ = my_func;
  flag_opacity_enrolled_ = true;
  return;
}


//--------------------------------------------------------------------------------------
//! \fn RayTracing::EnrollDensityFunction()
//! \brief Enrolls a function to calculate density.

void RayTracing::EnrollDensityFunction(RayTracingDensityFunc my_func) {
  UpdateDensity_ = my_func;
  flag_density_enrolled_ = true;
  return;
}


//--------------------------------------------------------------------------------------
//! \fn RayTracing::GetRadiationEnergyDensity()
//! \brief Returns radiation energy density array for output.

AthenaArray<Real> RayTracing::GetRadiationEnergyDensity(int ifr) const {
  AthenaArray<Real> e_rad_output(pmb->ncells3, pmb->ncells2, pmb->ncells1);
  for (int k=pmb->ks; k<=pmb->ke; ++k) {
    for (int j=pmb->js; j<=pmb->je; ++j) {
      for (int i=pmb->is; i<=pmb->ie; ++i) {
        e_rad_output(k,j,i) = rad_mom(IER_RT,k,j,i,ifr)/
          pmb->pmy_mesh->punit->speed_of_light_code;
      }
    }
  }
  return e_rad_output;
}

//--------------------------------------------------------------------------------------
//! \fn RayTracing::GetRadiationFluxDensity()
//! \brief Returns radiation flux density array for output.

AthenaArray<Real> RayTracing::GetRadiationFluxDensity(int ifr) const {
  AthenaArray<Real> flux_rad_output(3, pmb->ncells3, pmb->ncells2, pmb->ncells1);
  for (int k=pmb->ks; k<=pmb->ke; ++k) {
    for (int j=pmb->js; j<=pmb->je; ++j) {
      for (int i=pmb->is; i<=pmb->ie; ++i) {
        flux_rad_output(0,k,j,i) = rad_mom(IFX_RT,k,j,i,ifr);
        flux_rad_output(1,k,j,i) = rad_mom(IFY_RT,k,j,i,ifr);
        flux_rad_output(2,k,j,i) = rad_mom(IFZ_RT,k,j,i,ifr);
      }
    }
  }
  return flux_rad_output;
}


//--------------------------------------------------------------------------------------
//! \fn RayTracing::Prepare()
//! \brief Calculate opacity, initialize radiation arrays, etc. before ray tracing.

void RayTracing::Prepare() {
  rad_mom.ZeroClear();
  rad_mom0_fshld.ZeroClear();
  if (flag_opacity_enrolled_) {
    UpdateOpacity_(pmb, chi_rayt);
  } else {
    std::stringstream msg;
    msg << "### FATAL ERROR in RayTracing::Prepare" << std::endl
        << "Use EnrollOpacityFunction to set opacity." << std::endl;
    ATHENA_ERROR(msg);
  }
  if (flag_density_enrolled_) {
    UpdateDensity_(pmb, den_rayt);
  } else {
    // TODO(JGKIM): error if shielding calculation is turned on.
  }
  return;
}

int RayTracing::InteractWithCell(Photon *pphot, Real dlen) {
  int i = pphot->icell;
  int j = pphot->jcell;
  int k = pphot->kcell;
  int nfreq_extinct = 0;
  Real dlum_chi_dvol_inv[NFREQ];
  for (int f=0; f<NFREQ; f++) {
    if (pphot->lum[f] <= 0.0) {
      nfreq_extinct++;
      continue;
    }
    // Opacity per unit length
    Real chi = chi_rayt(k,j,i,f);
    Real dtau = chi*dlen;
    Real dlum = pphot->lum[f]*(1.0 - std::exp(-dtau));
    dlum_chi_dvol_inv[f] = dlum/(dvol_*chi);
    // Speed of light times radiation energy density
    rad_mom(IER_RT,k,j,i,f) += dlum_chi_dvol_inv[f];
    // Radiation flux density
    rad_mom(IFX_RT,k,j,i,f) += pphot->n1*dlum_chi_dvol_inv[f];
    rad_mom(IFY_RT,k,j,i,f) += pphot->n2*dlum_chi_dvol_inv[f];
    rad_mom(IFZ_RT,k,j,i,f) += pphot->n3*dlum_chi_dvol_inv[f];
    pphot->lum[f] -= dlum;
    pphot->tau[f] += dtau;

    if (pphot->tau[f] > tau_max_) pphot->lum[f] = 0.0;
  }

  return nfreq_extinct;
}


//--------------------------------------------------------------------------------------
//! \fn RayTracing::RayTracingDriver()
//! \brief Constructs a RayTracingDriver instance.

RayTracingDriver::RayTracingDriver(Mesh *pm, ParameterInput *pin) :
  pmy_mesh_(pm), comm_(pm, pin), point_src_list_(1),
  rotate_rays_(pin->GetOrAddBoolean("ray_tracing", "rotate_rays", true)),
  hpw_{rotate_rays_,
       pin->GetOrAddInteger("ray_tracing", "rseed", 0)},
  healpix_lev_min_(pin->GetOrAddInteger("ray_tracing", "healpix_lev_min", 4)),
  nexit_max_(pin->GetOrAddInteger("ray_tracing", "nexit_max", 50)),
  rays_per_cell_(pin->GetOrAddReal("ray_tracing", "rays_per_cell", 4.0)) {
  if (std::strcmp(COORDINATE_SYSTEM, "cartesian") != 0) {
    std::stringstream msg;
    msg << "### FATAL ERROR in RayTracingDriver::RayTracingDriver" << std::endl
        << "RayTracing only compatible with Cartesian coord." << std::endl;
      ATHENA_ERROR(msg);
  }

  if (pmy_mesh_->mesh_size.nx2==1 || pmy_mesh_->mesh_size.nx3==1) {
    std::stringstream msg;
    msg << "### FATAL ERROR in RayTracingDriver::RayTracingDriver" << std::endl
        << "RayTracing works only in 3D." << std::endl;
    ATHENA_ERROR(msg);
    return;
  }

  if (!(pmy_mesh_->use_uniform_meshgen_fn_[X1DIR]) ||
      !(pmy_mesh_->use_uniform_meshgen_fn_[X2DIR]) ||
      !(pmy_mesh_->use_uniform_meshgen_fn_[X3DIR])) {
    std::stringstream msg;
    msg << "### FATAL ERROR in RayTracingDriver constructor" << std::endl
        << "RayTracing only compatible with uniform mesh spacing." << std::endl;
      ATHENA_ERROR(msg);
  }

  if (rotate_rays_) {
    Pix2VecNest = &HEALPixWrapper::Pix2VecNestWithRotation;
  } else {
    Pix2VecNest = &HEALPixWrapper::Pix2VecNestWithoutRotation;
  }

  // Splitting distance needs to be calculated only once as long as Mesh::max_level
  // doesn't change during the simulation.
  CalculateSplitDistances();
  comm_.Initialize(0);
}

//--------------------------------------------------------------------------------------
//! \fn RayTracingDriver::~RayTracingDriver()
//! \brief Destroys a RayTracingDriver instance.

RayTracingDriver::~RayTracingDriver() {
  delete[] dist_split_;
}


//--------------------------------------------------------------------------------------
//! \fn RayTracingDriver::~RayTracingDriver()
//! \brief Initializer

void RayTracingDriver::Initialize(int reinit_flag) {
  comm_.Initialize(reinit_flag);
  return;
}

//--------------------------------------------------------------------------------------
//! \fn RayTracingDriver::AddPointSourceRadiator()
//! \brief Public function for adding a point source radiator to point_src_list.

void RayTracingDriver::AddPointSourceRadiator(PointSourceRadiator src) {
  point_src_list_.Push(&src);
  return;
}


//--------------------------------------------------------------------------------------
//! \fn RayTracingDriver::ResetPointSourceList()
//! \brief Public function for clearing point source radiator list

void RayTracingDriver::ResetPointSourceList() {
  point_src_list_.Reset();
  return;
}


//--------------------------------------------------------------------------------------
//! \fn RayTracingDriver::CalculateSplitDistances()
//! \brief Calculate 1d array storing splitting distances

void RayTracingDriver::CalculateSplitDistances() {
  // For multilevel implementation, need to make sure that dx_root is correctly calculated
  // for every process
  MeshBlock *pmb = pmy_mesh_->my_blocks(0);
  lo_max_ = pmy_mesh_->max_level - pmy_mesh_->root_level;
  dist_split_ = new Real[MAX_HEALPIX_LEVEL + lo_max_];

  // TODO(JGKIM): make sure that dx1_root = dx2_root = dx3_root or use minimum
  // Can we make sure that this yields the correct result for all processes?
  Real dx_root = pmb->pcoord->dx1f(0)*
    std::pow(2.0,pmb->loc.level - pmy_mesh_->root_level);

  // dist_small needs to be a negligibly small fraction of dx_root and acceptably small
  // even for simulations with many levels of refinement.
  dist_small_ = dx_root*small_number;;

  // Calculate distances at which photons split (see Sec 2.1 in Kim, J.-G. et al. 2017)
  for (int n=0; n<MAX_HEALPIX_LEVEL + lo_max_; ++n) {
    dist_split_[n] = std::sqrt(3.0/(PI*rays_per_cell_))*
      std::pow(2.0,n - lo_max_)*dx_root;
  }

  // For debugging
  // if (Globals::my_rank == 0) PrintSplittingDistances();

  return;
}


//--------------------------------------------------------------------------------------
//! \fn RayTracingDriver::Inject()
//! \brief Inject photons at the source positions

bool RayTracingDriver::Inject(const PointSourceRadiator& src) {
  // Number of rays at HEALPix level l: 12*4^l
  std::uint64_t nray = static_cast<std::uint64_t>(nray_base)*
    (1ULL << (2ULL*static_cast<std::uint64_t>(healpix_lev_min_)));
  // Photon phot(healpix_lev_min_, -1, src, 0.0);
  Photon phot = {};
  phot.lev = healpix_lev_min_;
  phot.mb_idx = -1;
  phot.x1_src = src.x1;
  phot.x2_src = src.x2;
  phot.x3_src = src.x3;

  // Find the meshblock to start in
  for (int b=0; b<pmy_mesh_->nblocal; ++b) {
    RegionSize& bsize = pmy_mesh_->my_blocks(b)->block_size;
    if ((phot.x1_src >= bsize.x1min) && (phot.x1_src < bsize.x1max) &&
        (phot.x2_src >= bsize.x2min) && (phot.x2_src < bsize.x2max) &&
        (phot.x3_src >= bsize.x3min) && (phot.x3_src < bsize.x3max)) {
      phot.mb_idx = b;
      // Find the cell to start in
      MeshBlock *pmb = pmy_mesh_->my_blocks(b);
      Real dx1i = 1.0/pmb->pcoord->dx1f(0);
      Real dx2i = 1.0/pmb->pcoord->dx2f(0);
      Real dx3i = 1.0/pmb->pcoord->dx3f(0);
      phot.icell = static_cast<int>(std::floor((phot.x1_src - bsize.x1min) * dx1i)) +
        pmb->is;
      phot.jcell = static_cast<int>(std::floor((phot.x2_src - bsize.x2min) * dx2i)) +
        pmb->js;
      phot.kcell = static_cast<int>(std::floor((phot.x3_src - bsize.x3min) * dx3i)) +
        pmb->ks;
    }
  }

  if (phot.mb_idx == -1) { // not a local source
    return false;
  }

  for (int f=0; f<NFREQ; f++) {
    phot.lum[f] = src.lum[f]/(static_cast<Real>(nray));
    phot.tau[f] = 0.0;
  }

  // Generate rays by 4^(healpix_lev_min) stride to facilitate parallelism.
  // For example, when healpix_lev_min = 2, HEALPpix id is generated in order
  // 0,16,...,16*11,16*12, 1,16+1,...,16*12+1, 2,16+2,...
  // where 12(=nray_base) is the number of rays at HEALPix base level 0.
  int stride = static_cast<int>(nray)/nray_base;
  int nside = 1 << healpix_lev_min_;
  Real vec[3];
  for (int m=0; m<stride; m++) {
    for (int l=0; l<nray_base; l++) {
      phot.levptr = l*stride + m;
      (hpw_.*Pix2VecNest)(nside, phot.levptr, vec);  // Determine ray direction
      phot.n1 = vec[0];
      phot.n2 = vec[1];
      phot.n3 = vec[2];

      // Prevent ray direction from becoming parallel to coordinate axes
      if (std::fabs(phot.n1) < small_number)
        phot.n1=phot.n1<0.0?-small_number:small_number;
      if (std::fabs(phot.n2) < small_number)
        phot.n2=phot.n2<0.0?-small_number:small_number;
      if (std::fabs(phot.n3) < small_number)
        phot.n3=phot.n3<0.0?-small_number:small_number;

      comm_.my_buf_.Push(&phot);
    }
  }
  return true;
}


//--------------------------------------------------------------------------------------
//! \fn RayTracingDriver::Prepare()
//! \brief Prepare to perform ray tracing.

void RayTracingDriver::Prepare() {
  // Update rotation matrix
  if (rotate_rays_) {
    hpw_.UpdateHEALPixRotationMatrix();
  }
  // Loop over point source list and inject photons and calculate the number of sources
  nsrc_tot_ = 0;
  nsrc_my_ = 0;
  PointSourceRadiator *point_src_arr = point_src_list_.GetArray();
  int size = point_src_list_.GetSize();
  for (int n=0; n<size; ++n) {
    if (Inject(point_src_arr[n])) nsrc_my_++;
  }

  for (int b=0; b<pmy_mesh_->nblocal; ++b) {
    MeshBlock *pmb = pmy_mesh_->my_blocks(b);
    pmb->prayt->Prepare();
  }

#ifdef MPI_PARALLEL
  MPI_Allreduce(&nsrc_my_, &nsrc_tot_, 1, MPI_INT, MPI_SUM, comm_.MPI_COMM_RAYT);
  // Compute maximum destroy count for bookkeeping
  comm_.ndest_max_ = static_cast<std::uint64_t>(nsrc_tot_)*
    static_cast<std::uint64_t>(nray_base)*
    (1ULL<<(2ULL*static_cast<std::uint64_t>(MAX_HEALPIX_LEVEL)));
#else
  nsrc_tot_ = nsrc_my_;
#endif
  return;
}


//--------------------------------------------------------------------------------------
//! \fn RayTracingDriver::WorkAfterRayTrace()
//! \brief Function called after RayTrace is finished.

void RayTracingDriver::WorkAfterRayTrace() {
  // TODO(JGKIM): Calculate escape fraction of radiation, etc.
  return;
}

//--------------------------------------------------------------------------------------
//! \fn RayTracingDriver::RayTrace()
//! \brief Perform one ray tracing.

void RayTracingDriver::RayTrace() {
  Prepare();
  if (Globals::nranks > 1) {
#ifdef MPI_PARALLEL
    comm_.InitializeCommVariablesAndPostIbcast();
    // Advance local photons and communicate until all done
    while (!comm_.all_done_) {
      // Traverse photons in my_buf until it is empty or a certain number of exiting
      // photons has reached.
      my_ndest_ = 0ULL;
      nexit_ = 0;
      while (nexit_ < nexit_max_) {
        if (comm_.my_buf_.GetSize() == 0) break;
        Traverse();
      }
      comm_.my_ndest_to_accum_ += my_ndest_;
      comm_.SendToNeighbors();
      comm_.ProbeAndReceive();
      comm_.CompleteSendRequests();
      if (comm_.my_buf_.GetSize() == 0)
        comm_.CheckAndUpdateDestroyCounter();
    }
#endif  // end of MPI_PARALLEL
  } else { // Process all local photons
    while (comm_.my_buf_.GetSize() != 0) {
      Traverse();
    }
  }

  // TODO(JGKIM): Need bug fix related to AddRadiationSource not working with AMR
  //  if (!pmy_mesh_->adaptive) {
    // Called in Mesh::LoadBalancingAndAdaptiveMeshRefinement if amr is turned on
    // Temporary solution..
  WorkAfterRayTrace();
    //}
  return;
}


//--------------------------------------------------------------------------------------
//! \fn RayTracingDriver::Traverse()
//! \brief Function to move one photon through grids.

void RayTracingDriver::Traverse() {
  RTNeighbor *pn;
  Photon *pphot = comm_.my_buf_.PopRear();

  int x1sign = (pphot->n1 > 0.0) ? 1 : -1;
  int x2sign = (pphot->n2 > 0.0) ? 1 : -1;
  int x3sign = (pphot->n3 > 0.0) ? 1 : -1;
  Real n1inv = 1.0/pphot->n1;
  Real n2inv = 1.0/pphot->n2;
  Real n3inv = 1.0/pphot->n3;

  MeshBlock *pmb = pmy_mesh_->my_blocks(pphot->mb_idx);
  RegionSize bsize = pmb->block_size;
  Coordinates *pcoord = pmb->pcoord;

  // Refinement level offset
  // lev_offset_max if current level == root_level
  //              0 if current level == max_level
  int lo = lo_max_ - (pmb->loc.level - pmy_mesh_->root_level);

  int edge = 0;
  int extinct = 0;
  while (!edge && !extinct) {
    Real rmin;
    int ox1 = 0;
    int ox2 = 0;
    int ox3 = 0;

    // Get face position
    int fidx1 = (x1sign==1) ? pphot->icell + 1 : pphot->icell;
    int fidx2 = (x2sign==1) ? pphot->jcell + 1 : pphot->jcell;
    int fidx3 = (x3sign==1) ? pphot->kcell + 1 : pphot->kcell;
    Real x1f = pcoord->x1f(fidx1);
    Real x2f = pcoord->x2f(fidx2);
    Real x3f = pcoord->x3f(fidx3);

    // Get distance from source to planes of intersection
    Real rx1 = std::fabs((x1f - pphot->x1_src)*n1inv);
    Real rx2 = std::fabs((x2f - pphot->x2_src)*n2inv);
    Real rx3 = std::fabs((x3f - pphot->x3_src)*n3inv);

    // Compute minimum distance and cell offset
    if (rx1 <= std::min(rx2,rx3)) {
      ox1 += x1sign;
      rmin = rx1;
    }
    if (rx2 <= std::min(rx1,rx3)) {
      ox2 += x2sign;
      rmin = rx2;
    }
    if (rx3 <= std::min(rx1,rx2)) {
      ox3 += x3sign;
      rmin = rx3;
    }

    if (DEBUG_RAYT) {
      // Add dist_small to handle a precision loss in floating point arithmetic
      if (rmin + dist_small_ < pphot->dist) {
        PrintPhotonInfo(*pphot, 0, 0, 0,  "Traverse: Check rmin!", 0);
        std::cout << "phot x1f,x2f,x3f: " << x1f << " " << x2f << " " << x3f << std::endl
                  << "rx1,rx2,rx3 " << rx1 << " " << rx2 << " " << rx3 << std::endl
                  << "rmin is smaller than dist!" << std::endl
                  << "rmin " << rmin << " dist " << pphot->dist << " dist_split "
                  << dist_split_[lo+pphot->lev] << std::endl;
        std::stringstream msg;
        msg << std::endl << "### FATAL ERROR in RayTracingDriver::Traverse" << std::endl;
        ATHENA_ERROR(msg);
      }
    }

    if (rmin > dist_split_[lo+pphot->lev]) {
      // Need to increase angular resolution by splitting
      Real dlen = dist_split_[lo+pphot->lev] - pphot->dist;
      if (dlen > 0.0) {
        pphot->dist = dist_split_[lo+pphot->lev];
        // Interact before splitting
        int nfreq_extinct = pmb->prayt->InteractWithCell(pphot, dlen);
        if (nfreq_extinct == NFREQ) {
          extinct = 1;
          continue;
        }
      }
      Split(pphot);
      return;
    } else {
      Real dlen = rmin - pphot->dist;
      pphot->dist = rmin;
      if (dlen > 0.0) {
        int nfreq_extinct = pmb->prayt->InteractWithCell(pphot, dlen);
        if (nfreq_extinct == NFREQ) {
          extinct = 1;
          continue;
        }
      }
    }

    // Change cell indices
    pphot->icell += ox1;
    pphot->jcell += ox2;
    pphot->kcell += ox3;

    // Calculate meshblock offset
    ox1 = ox2 = ox3 = 0;
    if (pphot->icell > pmb->ie) ox1++;
    else if (pphot->icell < pmb->is) ox1--;
    if (pphot->jcell > pmb->je) ox2++;
    else if (pphot->jcell < pmb->js) ox2--;
    if (pphot->kcell > pmb->ke) ox3++;
    else if (pphot->kcell < pmb->ks) ox3--;

    // Still within the same meshblock
    if ((ox1 == 0) && (ox2 == 0) && (ox3 == 0))
      continue;

    // If we are here, exiting current meshblock
    int nlev = pmb->pbval->nblevel[ox3+1][ox2+1][ox1+1];
    if (nlev == -1) { // exiting global domain
      edge = 1;
      break;
    }

    if (nlev != pmb->loc.level) {
      if (DEBUG_RAYT) {
        if (!pmy_mesh_->multilevel) {
          std::stringstream msg;
          msg << std::endl << "### FATAL ERROR in RayTracingDriver::Traverse" << std::endl
              << "Multilevel is turned off but nblevel is different!" << std::endl;
        ATHENA_ERROR(msg);
        }
      }
      if (nlev > pmb->loc.level) { // neighbor meshblock at finer level
        // calculate two integers (0 or 1) identifying refined neighbor and save to last
        // two bits of an integer variable
        int j = 0, fi = 0;
        if (ox1 == 0) {
          j++;
          Real x1 = pphot->x1_src + pphot->dist*pphot->n1;
          Real dx1i = 1.0/pmb->pcoord->dx1f(0);
          pphot->icell = static_cast<int>(std::floor((x1 - bsize.x1min) * (2.0*dx1i)));
          if (pphot->icell >= bsize.nx1) {
            fi = 1;
            pphot->icell -= bsize.nx1;
          } else {
            fi = 0;
          }
          pphot->icell += pmb->is;
        } else {
          pphot->icell -= bsize.nx1*ox1;
        }
        if (ox2 == 0) {
          j++;
          Real x2 = pphot->x2_src + pphot->dist*pphot->n2;
          Real dx2i = 1.0/pmb->pcoord->dx2f(0);
          pphot->jcell = static_cast<int>(std::floor((x2 - bsize.x2min) * (2.0*dx2i)));
          if (pphot->jcell >= bsize.nx2) {
            fi = (fi << 1) | 1;
            pphot->jcell -= bsize.nx2;
          } else {
            fi = fi << 1;
          }
          pphot->jcell += pmb->js;
        } else {
          pphot->jcell -= bsize.nx2*ox2;
        }
        if (ox3 == 0) {
          j++;
          Real x3 = pphot->x3_src + pphot->dist*pphot->n3;
          Real dx3i = 1.0/pmb->pcoord->dx3f(0);
          pphot->kcell = static_cast<int>(std::floor((x3 - bsize.x3min) * (2.0*dx3i)));
          if (pphot->kcell >= bsize.nx3) {
            fi = (fi << 1) | 1;
            pphot->kcell -= bsize.nx3;
          } else {
            fi = fi << 1;
          }
          pphot->kcell += pmb->ks;
        } else {
          pphot->kcell -= bsize.nx3*ox3;
        }
        if (j == 1) fi <<= 1;
        else if (j == 0) fi = 0; // set to zero for corner neighbor
        int bufid = ((ox1+1) << 6) | ((ox2+1) << 4) | ((ox3+1) << 2) | fi;
        pn = &comm_.neighbor_(pphot->mb_idx, comm_.bufidtonid_(pphot->mb_idx, bufid));
        pphot->mb_idx = pn->pnb->snb.lid;
        if (Globals::my_rank == pn->pnb->snb.rank) { // another local meshblock
          pmb = pmy_mesh_->my_blocks(pphot->mb_idx);
          bsize = pmb->block_size;
          pcoord = pmb->pcoord;
          lo--;
          continue;
        } else {
#ifdef MPI_PARALLEL
        edge = 2;
        break;
#else
        // If this is without MPI, something's wrong...
        std::stringstream msg;
        msg << std::endl << "### FATAL ERROR in RayTracingDriver::Traverse" << std::endl
            << "Cannot find the neighbor (finer) MeshBlock." << std::endl;
        ATHENA_ERROR(msg);
#endif
        }
      } else { // neighbor meshblock at coarser level
        // update cell indices
        int myfx1 = ((pmb->loc.lx1 & 1LL) == 1LL);
        int myfx2 = ((pmb->loc.lx2 & 1LL) == 1LL);
        int myfx3 = ((pmb->loc.lx3 & 1LL) == 1LL);
        if (ox1 == 0) {
          pphot->icell = (pphot->icell - pmb->is)/2 + myfx1*(bsize.nx1/2) + pmb->is;
        } else {
          pphot->icell -= ox1*(1 + (-ox1+1)/2 + ox1*myfx1)*(bsize.nx1/2);
        }
        if (ox2 == 0) {
          pphot->jcell = (pphot->jcell - pmb->js)/2 + myfx2*(bsize.nx2/2) + pmb->js;
        } else {
          pphot->jcell -= ox2*(1 + (-ox2+1)/2 + ox2*myfx2)*(bsize.nx2/2);
        }
        if (ox3 == 0) {
          pphot->kcell = (pphot->kcell - pmb->ks)/2 + myfx3*(bsize.nx3/2) + pmb->ks;
        } else {
          pphot->kcell -= ox3*(1 + (-ox3+1)/2 + ox3*myfx3)*(bsize.nx3/2);
        }
        int bufid = ((ox1+1) << 6) | ((ox2+1) << 4) | ((ox3+1) << 2);
        pn = &comm_.neighbor_(pphot->mb_idx, comm_.bufidtonid_(pphot->mb_idx, bufid));
        pphot->mb_idx = pn->pnb->snb.lid;
        if (Globals::my_rank == pn->pnb->snb.rank) { // another local meshblock
          pmb = pmy_mesh_->my_blocks(pphot->mb_idx);
          bsize = pmb->block_size;
          pcoord = pmb->pcoord;
          lo++;
          continue;
        } else {
#ifdef MPI_PARALLEL
        edge = 2;
        break;
#else
        // If this is without MPI, something's wrong...
        std::stringstream msg;
        msg << std::endl << "### FATAL ERROR in RayTracingDriver::Traverse" << std::endl
            << "Cannot find the neighbor (coarser) MeshBlock." << std::endl;
        ATHENA_ERROR(msg);
#endif
        }
      }
    } else { // neighbor meshblock at the same level
      // pn = &comm_.neighbor_(pphot->mb_idx, ox1+1, ox2+1, ox3+1);
      int bufid = ((ox1+1) << 6) | ((ox2+1) << 4) | ((ox3+1) << 2);
      pn = &comm_.neighbor_(pphot->mb_idx, comm_.bufidtonid_(pphot->mb_idx, bufid));
      if (DEBUG_RAYT) {
        if (pn->pnb == nullptr) {
          std::stringstream msg;
          msg << std::endl << "### FATAL ERROR in RayTracingDriver::Traverse" << std::endl
              << "Edge condition already checked!" << std::endl;
          ATHENA_ERROR(msg);
        }
      }
      // neighbor meshblock exists
      pphot->mb_idx = pn->pnb->snb.lid;
      pphot->icell -= bsize.nx1*ox1;
      pphot->jcell -= bsize.nx2*ox2;
      pphot->kcell -= bsize.nx3*ox3;
      if (Globals::my_rank == pn->pnb->snb.rank) { // another local meshblock
        pmb = pmy_mesh_->my_blocks(pphot->mb_idx);
        bsize = pmb->block_size;
        pcoord = pmb->pcoord;
        // No need to update level offset
        //lo = lo_max_ - (pmb->loc.level - pmy_mesh_->root_level)
      } else { // need to be passed to another process
#ifdef MPI_PARALLEL
        edge = 2;
#else
        // If this is without MPI, something's wrong...
        std::stringstream msg;
        msg << std::endl << "### FATAL ERROR in RayTracingDriver::Traverse" << std::endl
            << "Cannot find the neighbor MeshBlock." << std::endl;
        ATHENA_ERROR(msg);
#endif
      } // end of if (pn->pnb == nullptr) else ..
    } // end of if ((nlev != pmb->loc.level) && (nlev != -1)) else ..
  } // end of while (!edge && !extinct) ..

#ifdef MPI_PARALLEL
  if (edge == 1 || extinct) {
    my_ndest_ += 1ULL<<
      (2ULL*(static_cast<std::uint64_t>(MAX_HEALPIX_LEVEL - pphot->lev)));
  } else if (edge == 2) {
    // Store to exit_buf
    comm_.exit_buf_(comm_.ngbrlist_[pn->pnb->snb.rank]).Push(pphot);
    nexit_++;
  }
#endif

  return;
}

//--------------------------------------------------------------------------------------
//! \fn RayTracingDriver::Split()
//! \brief Function to split one photon into four children

void RayTracingDriver::Split(Photon *pphot) {
  RTNeighbor *pn;
  Photon phot = *pphot; // parent
  Photon photc = phot;  // children
  MeshBlock *pmb = pmy_mesh_->my_blocks(phot.mb_idx);
  RegionSize &bsize = pmb->block_size;
  Real dx1i = 1.0/pmb->pcoord->dx1f(0);
  Real dx2i = 1.0/pmb->pcoord->dx2f(0);
  Real dx3i = 1.0/pmb->pcoord->dx3f(0);
  photc.lev++;

  for (int f=0; f<NFREQ; ++f) {
    photc.lum[f] *= 0.25;
  }

  int ox1, ox2, ox3;
  int nside = 1 << photc.lev;
  Real vec[3];
  for (int m=0; m<4; ++m) {
    ox1 = ox2 = ox3 = 0;
    photc.levptr = 4*phot.levptr + m;
    photc.mb_idx = phot.mb_idx;
    (hpw_.*Pix2VecNest)(nside, photc.levptr, vec);
    photc.n1 = vec[0];
    photc.n2 = vec[1];
    photc.n3 = vec[2];

    if (std::fabs(photc.n1) < small_number)
      photc.n1 = photc.n1 < 0.0 ? -small_number : small_number;
    if (std::fabs(photc.n2) < small_number)
      photc.n2 = photc.n2 < 0.0 ? -small_number : small_number;
    if (std::fabs(photc.n3) < small_number)
      photc.n3 = photc.n3 < 0.0 ? -small_number : small_number;

    // Starting position
    Real x1_0 = photc.x1_src + photc.dist*photc.n1;
    Real x2_0 = photc.x2_src + photc.dist*photc.n2;
    Real x3_0 = photc.x3_src + photc.dist*photc.n3;

    // Find the cell to start in
    photc.icell = static_cast<int>(std::floor((x1_0 - bsize.x1min) * dx1i)) + pmb->is;
    photc.jcell = static_cast<int>(std::floor((x2_0 - bsize.x2min) * dx2i)) + pmb->js;
    photc.kcell = static_cast<int>(std::floor((x3_0 - bsize.x3min) * dx3i)) + pmb->ks;

    if (photc.icell > pmb->ie) ox1++;
    else if (photc.icell < pmb->is) ox1--;
    if (photc.jcell > pmb->je) ox2++;
    else if (photc.jcell < pmb->js) ox2--;
    if (photc.kcell > pmb->ke) ox3++;
    else if (photc.kcell < pmb->ks) ox3--;

    // This child belongs to current meshblock
    if ((ox1 == 0) && (ox2 == 0) && (ox3 == 0)) {
      comm_.my_buf_.Push(&photc);
      continue;
    }

    // If we are here, this child exits current meshblock
    int nlev = pmb->pbval->nblevel[ox3+1][ox2+1][ox1+1];
    if (nlev == -1) { // destroy
#ifdef MPI_PARALLEL
      my_ndest_ += 1ULL<<
        (2ULL*(static_cast<std::uint64_t>(MAX_HEALPIX_LEVEL - photc.lev)));
#endif
      continue;
    }

    if (nlev > pmb->loc.level) { // neighbor meshblock at finer level
      int j = 0, fi = 0;
      if (ox1 == 0) {
        j++;
        Real x1mid = 0.5*(bsize.x1min + bsize.x1max);
        if (x1_0 < x1mid) {
          fi = 0;
          photc.icell = static_cast<int>(std::floor((x1_0 - bsize.x1min) *
                                                    (2.0*dx1i))) + pmb->is;
        } else {
          fi = 1;
          photc.icell = static_cast<int>(std::floor((x1_0 - x1mid) *
                                                    (2.0*dx1i))) + pmb->is;
        }
      } else {
        photc.icell -= bsize.nx1*ox1;
      }
      if (ox2 == 0) {
        j++;
        Real x2mid = 0.5*(bsize.x2min + bsize.x2max);
        if (x2_0 < x2mid) {
          fi = fi << 1;
          photc.jcell = static_cast<int>(std::floor((x2_0 - bsize.x2min) *
                                                    (2.0*dx2i))) + pmb->js;
        } else {
          fi = (fi << 1) | 1;
          photc.jcell = static_cast<int>(std::floor((x2_0 - x2mid) *
                                                    (2.0*dx2i))) + pmb->js;
        }
      } else {
        photc.jcell -= bsize.nx2*ox2;
      }
      if (ox3 == 0) {
        j++;
        Real x3mid = 0.5*(bsize.x3min + bsize.x3max);
        if (x3_0 < x3mid) {
          fi = fi << 1;
          photc.kcell = static_cast<int>(std::floor((x3_0 - bsize.x3min) *
                                                    (2.0*dx3i))) + pmb->ks;
        } else {
          fi = (fi << 1) | 1;
          photc.kcell = static_cast<int>(std::floor((x3_0 - x3mid) *
                                                    (2.0*dx3i))) + pmb->ks;
        }
      } else {
        photc.kcell -= bsize.nx3*ox3;
      }
      if (j == 1) fi <<= 1;
      else if (j == 0) fi = 0; // set to zero for corner neighbor
      int bufid = ((ox1+1) << 6) | ((ox2+1) << 4) | ((ox3+1) << 2) | fi;
      pn = &comm_.neighbor_(photc.mb_idx, comm_.bufidtonid_(photc.mb_idx, bufid));
      photc.mb_idx = pn->pnb->snb.lid;
      if (Globals::my_rank == pn->pnb->snb.rank) { // another local meshblock
        comm_.my_buf_.Push(&photc);
        continue;
      } else {
#ifdef MPI_PARALLEL
        comm_.exit_buf_(comm_.ngbrlist_[pn->pnb->snb.rank]).Push(&photc);
        nexit_++;
        continue;
#endif
      }
    } else if (nlev < pmb->loc.level) { // neighbor meshblock at coarser level
      int bufid = ((ox1+1) << 6) | ((ox2+1) << 4) | ((ox3+1) << 2);
      pn = &comm_.neighbor_(photc.mb_idx, comm_.bufidtonid_(photc.mb_idx, bufid));
      photc.mb_idx = pn->pnb->snb.lid;
      // update cell indices
      int myfx1 = ((pmb->loc.lx1 & 1LL) == 1LL);
      int myfx2 = ((pmb->loc.lx2 & 1LL) == 1LL);
      int myfx3 = ((pmb->loc.lx3 & 1LL) == 1LL);
      if (ox1 == 0) {
        photc.icell = (photc.icell - pmb->is)/2 + myfx1*(bsize.nx1/2) + pmb->is;
      } else {
        photc.icell -= ox1*(1 + (-ox1+1)/2 + ox1*myfx1)*(bsize.nx1/2);
      }
      if (ox2 == 0) {
        photc.jcell = (photc.jcell - pmb->js)/2 + myfx2*(bsize.nx2/2) + pmb->js;
      } else {
        photc.jcell -= ox2*(1 + (-ox2+1)/2 + ox2*myfx2)*(bsize.nx2/2);
      }
      if (ox3 == 0) {
        photc.kcell = (photc.kcell - pmb->ks)/2 + myfx3*(bsize.nx3/2) + pmb->ks;
      } else {
        photc.kcell -= ox3*(1 + (-ox3+1)/2 + ox3*myfx3)*(bsize.nx3/2);
      }
      if (Globals::my_rank == pn->pnb->snb.rank) { // another local meshblock
        comm_.my_buf_.Push(&photc);
        continue;
      } else {
#ifdef MPI_PARALLEL
        comm_.exit_buf_(comm_.ngbrlist_[pn->pnb->snb.rank]).Push(&photc);
        nexit_++;
        continue;
#endif
      }
    } else { // neighbor meshblock at the same level
      int bufid = ((ox1+1) << 6) | ((ox2+1) << 4) | ((ox3+1) << 2);
      pn = &comm_.neighbor_(photc.mb_idx, comm_.bufidtonid_(photc.mb_idx, bufid));
      photc.mb_idx = pn->pnb->snb.lid;
      photc.icell -= bsize.nx1*ox1;
      photc.jcell -= bsize.nx2*ox2;
      photc.kcell -= bsize.nx3*ox3;
      if (Globals::my_rank == pn->pnb->snb.rank) { // another local meshblock
        comm_.my_buf_.Push(&photc);
        continue;
      } else {
#ifdef MPI_PARALLEL
        comm_.exit_buf_(comm_.ngbrlist_[pn->pnb->snb.rank]).Push(&photc);
        nexit_++;
        continue;
#endif
      }
    }

    if (DEBUG_RAYT) {
      // We shouldn't be here..something went wrong..
      std::stringstream msg;
      msg << std::endl << "### FATAL ERROR in RayTracingDriver::Split" << std::endl
          << "Reached the end of Split..something went wrong.." << std::endl;
      ATHENA_ERROR(msg);
    }
  }
  return;
}

//--------------------------------------------------------------------------------------
//! \fn RayTracingDriver::PrintSplittingDistances()
//! \brief Function to print splitting distances for debugging.

void RayTracingDriver::PrintSplittingDistances() {
  MeshBlock *pmb = pmy_mesh_->my_blocks(0);
  Real dx_root = pmb->pcoord->dx1f(0)*
    std::pow(2.0,pmb->loc.level - pmy_mesh_->root_level);

  std::cout << "lo_max root_level dx_root "
            << lo_max_ << " " << pmy_mesh_->root_level
            << " " << dx_root << std::endl;
  for (int lo=0; lo<=lo_max_; ++lo) {
    std::cout << "splitting distance at level offset " << lo << std::endl;
    for (int l=0; l<MAX_HEALPIX_LEVEL; ++l) {
      std::cout << dist_split_[lo+l] << " ";
    }
    std::cout << std::endl;
  }
  return;
}


//--------------------------------------------------------------------------------------
//! \fn RayTracingDriver::PrintNeighborBlockInfo()
//! \brief Function to print neighbor meshblock information for debugging.

void RayTracingDriver::PrintNeighborBlockInfo(NeighborBlock nb) {
  SimpleNeighborBlock snb = nb.snb;
  NeighborIndexes ni = nb.ni;
  std::cout << "snb (rank,lev,lid,gid) " << snb.rank << " " << snb.level << " "
            << snb.lid << " " << snb.gid << std::endl;
  std::cout << "ni (ox1,ox2,ox3,fi1,fi2) " << ni.ox1 << " " << ni.ox2 << " "
            << ni.ox3 << " " << ni.fi1 << " " << ni.fi2 << " "
            << "NeighborConnect type " << static_cast<int>(ni.type) << std::endl;
  std::cout << "bufid, eid, targetid, fid " << nb.bufid << " " << nb.eid << " "
            << nb.targetid << " " << nb.fid << std::endl;
  return;
}

//--------------------------------------------------------------------------------------
//! \fn RayTracingDriver::PrintPhotonInfo()
//! \brief Function to print photon information for debugging.

void RayTracingDriver::PrintPhotonInfo(Photon phot,
                                       int ox1, int ox2, int ox3,
                                       std::string str, int err_flag) {
  MeshBlock *pmb = pmy_mesh_->my_blocks(phot.mb_idx);
  RegionSize &bsize = pmb->block_size;
  LogicalLocation loc = pmy_mesh_->my_blocks(phot.mb_idx)->loc;
  int lo = lo_max_ - (pmb->loc.level - pmy_mesh_->root_level);
  Real x1_0 = phot.x1_src + phot.dist*phot.n1;
  Real x2_0 = phot.x2_src + phot.dist*phot.n2;
  Real x3_0 = phot.x3_src + phot.dist*phot.n3;
  std::stringstream msg;
  msg << std::endl << "### Photon Info: " << str << std::endl
#ifdef MPI_PARALLEL
      << "my_rank: " << Globals::my_rank << std::endl
#endif
      << "phot mb_idx, lev, levptr: " << phot.mb_idx << " "
      << phot.lev << " " << phot.levptr << std::endl
      << "mb lev lx1 lx2 lx3: " << loc.level << " "
      << loc.lx1 << " " << loc.lx2 << " " << loc.lx3 << std::endl
      << "mb region x1min x1max: " << bsize.x1min << " " << bsize.x1max << std::endl
      << "mb region x2min x2max: " << bsize.x2min << " " << bsize.x2max << std::endl
      << "mb region x3min x3max: " << bsize.x3min << " " << bsize.x3max << std::endl
      << "mb is js ks: " << pmb->is << " " << pmb->js << " " << pmb->ks << std::endl
      << "mb ie je ke: " << pmb->ie << " " << pmb->je << " " << pmb->ke << std::endl
      << "phot icell,jcell,kcell: " << phot.icell << " "
      << phot.jcell << " " << phot.kcell << std::endl
      << "phot x1_0,x2_0,x3_0: " << std::scientific << std::setprecision(16)
      << x1_0 << " " << x2_0 << " " << x3_0 << std::endl
      << "phot n1,n2,n3: " << phot.n1 << " " << phot.n2 << " " << phot.n3 << std::endl
      << "ox1,ox2,ox3: " << ox1 << " " << ox2 << " " << ox3 << std::endl
      << "level_offset, dist, dist_split: " << lo << " "
      << phot.dist << " " << dist_split_[lo+phot.lev] << std::endl;

  if (err_flag) {
    ATHENA_ERROR(msg);
  } else {
    std::cout << msg.str();
  }
  return;
}
