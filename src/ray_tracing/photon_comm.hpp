#ifndef RAY_TRACING_PHOTON_COMM_HPP_
#define RAY_TRACING_PHOTON_COMM_HPP_
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file photon_comm.hpp
//! \brief Provides PhotonCommunicator class template for communicating photons
//!
//!=======================================================================================

// C headers

// C++ headers
#include <sstream>
#include <string>  // string

// Athena++ headers
#include "../athena.hpp"
#include "../bvals/bvals_interfaces.hpp"
#include "../mesh/mesh.hpp"
#include "circular_buffer.hpp"

#define DEBUG_RAYT_COMM 0

struct RTNeighbor {
  NeighborBlock *pnb;
  MeshBlock *pmb;

  RTNeighbor() : pnb(NULL), pmb(NULL) {}
};

//--------------------------------------------------------------------------------------
//! \class PhotonCommunicator
//! \brief Defines the class for communicating photons

template<typename T>
class PhotonCommunicator {
 public:
  PhotonCommunicator(Mesh *pmesh, ParameterInput *pin);
  ~PhotonCommunicator();

  Mesh *pmy_mesh_;
  CircularBuffer<T> my_buf_;
  AthenaArray<RTNeighbor> neighbor_;   //!> links to neighbors
  AthenaArray<int> bufidtonid_;

  //! Maximum number of neighboring meshblocks in 3D (BoundaryBase::nneighbor)
  static constexpr int nngbr_mb_max_ = 56;
  //! Maximum value for buffer id: 168 = 2^7 + 2^5 + 2^3 for ox1=ox2=ox3=1 and fi1=fi2=0
  //! See BoundaryBase::CreateBufferID()
  static constexpr int bufid_max_ = 168;

  void Initialize(int reinit_flag);
  void CreateNeighborRankList();
  void CreateNeighborRankList(MeshBlock *pmb, int mb_idx);
  void ClearNeighborRankList();
#ifdef MPI_PARALLEL
  void CreateBuffers(int init_flag);
  void ClearBuffers();
  void CreateMPIWindow();
  void DestroyMPIWindow();
  void InitializeCommVariablesAndPostIbcast();
  void SendToNeighbors();
  void ProbeAndReceive();
  void CompleteSendRequests();
  void CheckAndUpdateDestroyCounter();

  MPI_Comm MPI_COMM_RAYT;
  int tag_rayt_;
  int nngbr_; //!> Number of neighbor "ranks"
  int nsendproc_;
  int *ranklist_, *ngbrlist_; //!> mapping from rank to neighbor id, and vice versa

  // Buffers for MPI communication
  AthenaArray<CircularBuffer<T>> exit_buf_;  //!> List of exiting photons
  byte **send_buf_;
  byte *recv_buf_;
  int *send_buf_max_byte_;
  int recv_buf_max_byte_; //!> Current buffer sizes in byte

  int *nidsenddone_;
  MPI_Request *send_req_;

  // Variables for destroy count update with remote memory access operations
  int all_done_, all_done_others_, flag_all_done_;
  MPI_Request all_done_req_;
  MPI_Win win_;
  MPI_Info win_info_;
  std::uint64_t *destcnt_, ndest_max_, my_ndest_to_accum_;
#endif
};

#endif  // RAY_TRACING_PHOTON_COMM_HPP_
