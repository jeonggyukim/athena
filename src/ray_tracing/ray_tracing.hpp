#ifndef RAY_TRACING_RAY_TRACING_HPP_
#define RAY_TRACING_RAY_TRACING_HPP_
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file ray_tracing.hpp
//! \brief definitions for RayTracing and RayTracing driver classes

// There are generalizations to be considered in several directions.
// 1 - Different types of Photon : simple, ncr chemistry, simple column density
//     calculation, etc.
//     Accordingly, InteractWithCell should support simple rad_mom as well as shielding
//     function calculations.
// 2. RayTracingDriver may contain multiple Photon containers.
// 3. Photon injection should support multiple types: point source, plane-parallel.
//    Accordingly, Photon should contain different information depending on the
//    source type

#define DEBUG_RAYT 0

// C headers

// C++ headers
#include <algorithm>
#include <random>  // mt19937, uniform_real_distribution
#include <string>

// Athena++ headers
#include "../athena.hpp"
#include "../globals.hpp"
#include "circular_buffer.hpp"
#include "photon.hpp"
#include "photon_comm.hpp"

// MPI headers
#ifdef MPI_PARALLEL
#include <mpi.h>
#endif

class Mesh;
class MeshBlock;
class ParameterInput;
class RayTracingDriver;

enum RaytIndex {IER_RT=0, IFX_RT=1, IFY_RT=2, IFZ_RT=3};

// TODO(JGKIM): can we make this more flexible (without performance penalty)?
constexpr int NFREQ = NFREQ_RAYT;

struct PointSourceRadiator {
  Real x1, x2, x3;     //!> source position
  Real lum[NFREQ];     //!> luminosity
  // NOTE: photon number density should also be followed for photochemistry
  // Real lum_phot[NFREQ];

  PointSourceRadiator() {}

  PointSourceRadiator(Real x1, Real x2, Real x3) :
    x1(x1), x2(x2), x3(x3) {}
};

// struct PPLocationAndDirection {
//   int mb_idx;       //!> local MeshBlock index
//   int icell, jcell, kcell; //!> cell indices in meshblock
//   Real n1, n2, n3; //!> unit vector specifying ray direction
// };

// struct PPPointSource {
//   int lev;          //!> HEALPix level
//   int levptr;       //!> HEALPix pixel number
//   Real x1_src, x2_src, x3_src; //!> source position
//   Real dist;       //!> distance from source to current position
// };

// template <int n>
// struct PPLuminosity {
//   Real lum[n]; //!> luminosity
//   Real tau[n]; //!> optical depth from source
// };

// Not an ideal solution, but works fine as long as we use CircularBuffer as a stack
using PointSourceRadiatorList = CircularBuffer<PointSourceRadiator>;

//--------------------------------------------------------------------------------------
//! \class HEALPixWrapper
//! \brief Defines the class for ray tracing owned by each meshblock.

class HEALPixWrapper {
 public:
  HEALPixWrapper(bool rotate_rays, int rseed);
  ~HEALPixWrapper();

  void Pix2VecNestWithoutRotation(int nside, int ipix, Real *vec);
  void Pix2VecNestWithRotation(int nside, int ipix, Real *vec);
  void UpdateHEALPixRotationMatrix();

 private:
  bool rotate_rays_;
  int rseed_;
  // Euler angles
  Real phi_euler_;    // rotation around the z-axis
  Real theta_euler_;  // rotation around the x'-axis
  Real psi_euler_;    // rotation around the z''-axis
  Real rotation_[3][3];
  std::mt19937_64 rng_generator_;
  std::uniform_real_distribution<Real> udist_;
};


//--------------------------------------------------------------------------------------
//! \class RayTracing
//! \brief Defines the class for ray tracing owned by each meshblock.

class RayTracing {
  friend RayTracingDriver;

 public:
  RayTracing(MeshBlock *pmy_block, ParameterInput *pin);
  ~RayTracing();

  // Radiation moments: (c Erad, Frad)
  AthenaArray<Real> rad_mom;
  // Opacity per length
  AthenaArray<Real> chi_rayt;

  // Pseudo-radiation density times c used for photochemistry
  AthenaArray<Real> rad_mom0_fshld;
  // Density used for column density calculation
  AthenaArray<Real> den_rayt;

  void EnrollOpacityFunction(RayTracingOpacityFunc my_func);
  void EnrollDensityFunction(RayTracingDensityFunc my_func);

  AthenaArray<Real> GetRadiationEnergyDensity(int ifr) const;
  AthenaArray<Real> GetRadiationFluxDensity(int ifr) const;

 private:
  int InteractWithCell(Photon *phot, Real dlen);
  void Prepare();

  MeshBlock* pmb;  //!> ptr to MeshBlock containing this RayTracing
  Real dvol_;      //!> cell volume in code unit
  Real tau_max_;   //!> Stop following a ray if tau > tau_max at all frequency bins
  static constexpr Real tau_max_def_ = 20.0;  //!> exp(-20) ~ 2.1e-9
  // Function pointer to update opacity for ray tracing
  RayTracingOpacityFunc UpdateOpacity_;
  RayTracingDensityFunc UpdateDensity_;
  bool flag_opacity_enrolled_;
  bool flag_density_enrolled_;
};


//--------------------------------------------------------------------------------------
//! \class RayTracingDriver
//! \brief Defines the class for ray tracing driver owned by each rank.

class RayTracingDriver {
 public:
  RayTracingDriver(Mesh *pm, ParameterInput *pin);
  ~RayTracingDriver();

  // NOTE: Ideally, this should be able to handle multiple types of photons
  // (plane-parallel, point source)
  void RayTrace();
  void Initialize(int reinit_flag);
  void AddPointSourceRadiator(PointSourceRadiator src);
  void ResetPointSourceList();
  void WorkAfterRayTrace();

 private:
  Mesh *pmy_mesh_;
  PhotonCommunicator<Photon> comm_;

  // NOTE: ART specific
  // Source list needs to be updated every time step
  PointSourceRadiatorList point_src_list_;
  bool rotate_rays_;
  HEALPixWrapper hpw_;

  // TODO(JGKIM): better to use enum for clarity?
  // enum BreakLoop {SEND=1, SPLIT=2, DESTROY=3};
  int nexit_max_; //!> Communicate photons if nexit >= nexit_max (MPI only)
  int nexit_; //!> Number of exiting photons

  // NOTE: ART specific
  int nsrc_my_;   //!> Number of point sources in local subdomain
  int nsrc_tot_;  //!> Total (global) number of point sources (for ndest_max calculation)
  int healpix_lev_min_; //!> HEALPix level of injected photons
  int lo_max_;    //!> maximum value of level offset (= max_level - root_level)
  Real rays_per_cell_; //!> Nominal angular resolution parameter
  Real dist_small_; //!> Tolerance for distance to cell calculation
  Real *dist_split_; //!> Splitting distances as a function of (healpix_level - ref_level)

  void Prepare();

  // Ray Tracer
  // NOTE: ART specific
  // NOTE: splitting position depends on source type;
  // Photon initialization should also change..
  bool Inject(const PointSourceRadiator &src);
  void Traverse();
  void Split(Photon *pphot);
  // NOTE: only for point source
  void CalculateSplitDistances();
  // NOTE: only for point source
  using Pix2VecNestFunc = void (HEALPixWrapper::*)(int nside, int ipix, Real *vec);
  // NOTE: only for point source
  Pix2VecNestFunc Pix2VecNest;

#ifdef MPI_PARALLEL
  // NOTE: RayTracer
  std::uint64_t my_ndest_;
#endif

  // For debugging
  void PrintSplittingDistances();
  void PrintNeighborBlockInfo(NeighborBlock nb);
  void PrintPhotonInfo(Photon phot,
                       int ox1, int ox2, int ox3,
                       std::string str, int err_flag);
};

#endif  // RAY_TRACING_RAY_TRACING_HPP_
