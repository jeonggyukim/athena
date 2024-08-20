//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file photon_comm.cpp
//! \brief Implementation of functions in classes PhotonCommunicator
//========================================================================================

// Athena++ headers
#include "photon.hpp"
#include "photon_comm.hpp"

//--------------------------------------------------------------------------------------
//! \fn PhotonCommunicator::PhotonCommunicator(MeshBlock *pmb, ParameterInput *pin)
//! \brief Constructs a PhotonCommunicator instance.

template<typename T>
PhotonCommunicator<T>::PhotonCommunicator(Mesh *pm, ParameterInput *pin) :
  pmy_mesh_(pm), my_buf_(1) {
#ifdef MPI_PARALLEL
  MPI_Comm_dup(MPI_COMM_WORLD, &MPI_COMM_RAYT);
  // (TODO) This is an ad hoc tag declaration. It may not be an issue as ART communication
  // won't overlap with others.
  tag_rayt_ = 51234;
#endif
}


//--------------------------------------------------------------------------------------
//! \fn PhotonCommunicator::~PhotonCommunicator()
//! \brief Destroys a PhotonCommunicator instance.

template<typename T>
PhotonCommunicator<T>::~PhotonCommunicator() {
  ClearNeighborRankList();
#ifdef MPI_PARALLEL
  // Free receive buffer once
  std::free(recv_buf_);
  DestroyMPIWindow();
  MPI_Comm_free(&MPI_COMM_RAYT);
#endif
}


//--------------------------------------------------------------------------------------
//! \fn PhotonCommunicator::Initialize()
//! \brief Find neighbors and allocate memories.
//!
//! When AMR is turned on and meshblock tree is updated, it is called in
//! Mesh::RedistributeAndRefineMeshBlocks with non-zero reinit_flag.

template<typename T>
void PhotonCommunicator<T>::Initialize(int reinit_flag) {
  if (reinit_flag) ClearNeighborRankList();
  CreateNeighborRankList();
#ifdef MPI_PARALLEL
  CreateBuffers(reinit_flag);
  if (!reinit_flag) CreateMPIWindow();
#endif
  return;
}


//--------------------------------------------------------------------------------------
//! \fn PhotonCommunicator::CreateNeighborRankList()
//! \brief Create neighbor rank lists and buffers for all meshblocks.

template<typename T>
void PhotonCommunicator<T>::CreateNeighborRankList() {
  // Greedy allocation for worst-case of refined 3D
  neighbor_.NewAthenaArray(pmy_mesh_->nblocal, nngbr_mb_max_);
  bufidtonid_.NewAthenaArray(pmy_mesh_->nblocal, bufid_max_+1);
  for (int b=0; b<pmy_mesh_->nblocal; ++b)
    for (int i=0; i<=bufid_max_; ++i)
      bufidtonid_(b,i) = -1;

#ifdef MPI_PARALLEL
  nngbr_ = 0;
  ngbrlist_ = reinterpret_cast<int *>(std::malloc(Globals::nranks*sizeof(int)));
  for (int i=0; i<Globals::nranks; ++i)
    ngbrlist_[i] = -1;
#endif

  for (int b=0; b<pmy_mesh_->nblocal; ++b)
    CreateNeighborRankList(pmy_mesh_->my_blocks(b), b);

  // std::stringstream msg;
  // msg << "### Terminating..." << std::endl;
  // ATHENA_ERROR(msg);
  // #ifdef MPI_PARALLEL
  //   for (int n=0 ; n<nngbr_; n++) {
  //     std::cout << "My rank: " << Globals::my_rank << ", nngbr: " << nngbr_ << " ,"
  //               << n << "-th neighbor's rank : " << ranklist_[n] << std::endl;
  //   }
  // #endif

  return;
}


//--------------------------------------------------------------------------------------
//! \fn PhotonCommunicator::CreateNeighborRankList()
//! \brief Create a neighbor rank list for a meshblock.

template<typename T>
void PhotonCommunicator<T>::CreateNeighborRankList(MeshBlock *pmb, int mb_idx) {
  RTNeighbor *pn;
  // Save pointer to each neighbor
  for (int i=0; i<pmb->pbval->nneighbor; ++i) {
    NeighborBlock &nb = pmb->pbval->neighbor[i];
    NeighborIndexes& ni = nb.ni;
    int bufid = ((ni.ox1+1) << 6) | ((ni.ox2+1) << 4) | ((ni.ox3+1) << 2) |
      (ni.fi1 << 1) | ni.fi2;
    bufidtonid_(mb_idx,bufid) = i;
    pn = &neighbor_(mb_idx,i);
    pn->pnb = &nb;

    if (pmb->loc.level > pmb->pbval->nblevel[ni.ox3+1][ni.ox2+1][ni.ox1+1]) {
      // Coarser neighbor meshblock
      int bufid_, ox1_, ox2_, ox3_;
      int myfx1 = ((pmb->loc.lx1 & 1LL) == 1LL);
      int myfx2 = ((pmb->loc.lx2 & 1LL) == 1LL);
      int myfx3 = ((pmb->loc.lx3 & 1LL) == 1LL);
      if (ni.type == NeighborConnect::face) {
        // For face neighbor, register two edges and one corner as valid bufid
        // Can use BoundaryBase::CreateBufferID instead
        if (ni.ox1 != 0) {
          ox2_ = -2*myfx2 + 1;
          ox3_ = -2*myfx3 + 1;
          bufid_ = ((ni.ox1+1) << 6) | ((ox2_+1) << 4) | ((ox3_+1) << 2);
          bufidtonid_(mb_idx,bufid_) = i;
          bufid_ = ((ni.ox1+1) << 6) | ((ni.ox2+1) << 4) | ((ox3_+1) << 2);
          bufidtonid_(mb_idx,bufid_) = i;
          bufid_ = ((ni.ox1+1) << 6) | ((ox2_+1) << 4) | ((ni.ox3+1) << 2);
          bufidtonid_(mb_idx,bufid_) = i;
        } else if (ni.ox2 != 0) {
          ox1_ = -2*myfx1 + 1;
          ox3_ = -2*myfx3 + 1;
          bufid_ = ((ox1_+1) << 6) | ((ni.ox2+1) << 4) | ((ox3_+1) << 2);
          bufidtonid_(mb_idx,bufid_) = i;
          bufid_ = ((ox1_+1) << 6) | ((ni.ox2+1) << 4) | ((ni.ox3+1) << 2);
          bufidtonid_(mb_idx,bufid_) = i;
          bufid_ = ((ni.ox1+1) << 6) | ((ni.ox2+1) << 4) | ((ox3_+1) << 2);
          bufidtonid_(mb_idx,bufid_) = i;
        } else if (ni.ox3 != 0) {
          ox1_ = -2*myfx1 + 1;
          ox2_ = -2*myfx2 + 1;
          bufid_ = ((ox1_+1) << 6) | ((ox2_+1) << 4) | ((ni.ox3+1) << 2);
          bufidtonid_(mb_idx,bufid_) = i;
          bufid_ = ((ni.ox1+1) << 6) | ((ox2_+1) << 4) | ((ni.ox3+1) << 2);
          bufidtonid_(mb_idx,bufid_) = i;
          bufid_ = ((ox1_+1) << 6) | ((ni.ox2+1) << 4) | ((ni.ox3+1) << 2);
          bufidtonid_(mb_idx,bufid_) = i;
        }
      } else if (ni.type == NeighborConnect::edge) {
        // For edge neighbor, register one corner as valid bufid
        if (ni.ox1 == 0) {
          ox1_ = -2*myfx1 + 1;
          bufid_ = ((ox1_+1) << 6) | ((ni.ox2+1) << 4) | ((ni.ox3+1) << 2);
          bufidtonid_(mb_idx,bufid_) = i;
        } else if (ni.ox2 == 0) {
          ox2_ = -2*myfx2 + 1;
          bufid_ = ((ni.ox1+1) << 6) | ((ox2_+1) << 4) | ((ni.ox3+1) << 2);
          bufidtonid_(mb_idx,bufid_) = i;
        } else if (ni.ox3 == 0) {
          ox3_ = -2*myfx3 + 1;
          bufid_ = ((ni.ox1+1) << 6) | ((ni.ox2+1) << 4) | ((ox3_+1) << 2);
          bufidtonid_(mb_idx,bufid_) = i;
        }
      }
    }
#ifdef MPI_PARALLEL
    SimpleNeighborBlock& snb = nb.snb;
    // Determine the number of unique neighboring ranks: nngbr
    // Create an array whose n-th elementh gives MPI rank of n-th neighbor.
    if (snb.rank != Globals::my_rank) {
      int new_ngbr = 1;
      // Check if this rank is already in the list
      if (nngbr_ > 0) {
        for (int n=0; n<nngbr_ ; ++n) {
          if (snb.rank == ranklist_[n]) {
            new_ngbr = 0;
            break;
          }
        }
      }
      if (new_ngbr == 1) {
        ngbrlist_[snb.rank] = nngbr_;
        if (nngbr_ == 0) {
          ranklist_ = reinterpret_cast<int *>(std::malloc(sizeof(int)));
        } else {
          ranklist_ = reinterpret_cast<int *>
            (std::realloc(ranklist_, (nngbr_+1)*sizeof(int)));
        }
        ranklist_[nngbr_] = snb.rank;
        nngbr_++;
      }
    }
#endif
  }
  return;
}

//--------------------------------------------------------------------------------------
//! \fn PhotonCommunicator::ClearNeighborRankList()
//! \brief De-allocate memories for neighbor rank list and buffers

template<typename T>
void PhotonCommunicator<T>::ClearNeighborRankList() {
  neighbor_.DeleteAthenaArray();
  bufidtonid_.DeleteAthenaArray();
#ifdef MPI_PARALLEL
  nngbr_ = 0;
  std::free(ngbrlist_);
  std::free(ranklist_);
  ClearBuffers();
#endif
  return;
}

#ifdef MPI_PARALLEL
//--------------------------------------------------------------------------------------
//! \fn PhotonCommunicator::CreateBuffers()
//! \brief Create send receive buffers for communication.

template<typename T>
void PhotonCommunicator<T>::CreateBuffers(int reinit_flag) {
  if (reinit_flag == 0) {
    // Allocate memory for receive buffer
    recv_buf_max_byte_ = 2*sizeof(T);
    recv_buf_ = reinterpret_cast<byte *>(std::malloc(recv_buf_max_byte_));
  }
  // Allocate memory for exit_buf, send buffers, and send req handles
  exit_buf_.NewAthenaArray(nngbr_);
  send_buf_ = reinterpret_cast<byte **>(std::malloc(nngbr_*sizeof(byte *)));
  send_buf_max_byte_ = reinterpret_cast<int *>(malloc(nngbr_*sizeof(int)));
  for (int n=0; n<nngbr_; ++n) {
    send_buf_max_byte_[n] = 2*sizeof(T);
    send_buf_[n] = reinterpret_cast<byte *>(std::malloc(send_buf_max_byte_[n]));
  }
  send_req_ = new MPI_Request[nngbr_];
  nidsenddone_ = new int[nngbr_];
  return;
}

//--------------------------------------------------------------------------------------
//! \fn PhotonCommunicator::CreateBuffers()
//! \brief De-allocate memories for communication buffers

template<typename T>
void PhotonCommunicator<T>::ClearBuffers() {
  // No need to free recv buffer
  exit_buf_.DeleteAthenaArray();
  for (int n=0; n<nngbr_; ++n) std::free(send_buf_[n]);
  std::free(send_buf_);
  std::free(send_buf_max_byte_);
  delete [] nidsenddone_;
  delete [] send_req_;
  return;
}

//--------------------------------------------------------------------------------------
//! \fn PhotonCommunicator::InitializeCommVariablesAndPostIbcast()
//! \brief Initialize communication variables and post MPI_Ibcast (except root)

template<typename T>
void PhotonCommunicator<T>::InitializeCommVariablesAndPostIbcast() {
  int err;
  my_ndest_to_accum_ = 0ULL;
  *destcnt_ = 0ULL;
  nsendproc_ = 0;
  for (int p=0; p<nngbr_; p++) {
    send_req_[p] = MPI_REQUEST_NULL;
    nidsenddone_[p] = 0;
  }
  // Initialize variables for signaling all_done
  all_done_ = 0;
  all_done_req_ = MPI_REQUEST_NULL;
  flag_all_done_ = 0;
  if (Globals::my_rank != 0) {
    err = MPI_Ibcast(&all_done_others_, 1, MPI_INT, 0, MPI_COMM_RAYT, &all_done_req_);
    if (DEBUG_RAYT_COMM) {
      if (err) {
        std::stringstream msg;
        msg << "### FATAL ERROR in PhotonCommunicator::RayTrace" << std::endl
            << "Rank " << Globals::my_rank
            << " MPI_Ibcast failed with err : " << err << std::endl;
        ATHENA_ERROR(msg);
      }
    }
  }
  // Barrier synchronization before entering loop
  MPI_Barrier(MPI_COMM_RAYT);
  return;
}


//--------------------------------------------------------------------------------------
//! \fn PhotonCommunicator::SendToNeighbors()
//! \brief Send photons to neighbor processes.

template<typename T>
void PhotonCommunicator<T>::SendToNeighbors() {
  // Use two-sided communications to tranfer photons. Loop over neighbors and Isend
  // photons in exit_buf.
  for (int n=0; n<nngbr_; ++n) {
    if (exit_buf_(n).GetSize() != 0 && send_req_[n] == MPI_REQUEST_NULL) {
      int nsendbyte = exit_buf_(n).GetSize()*sizeof(T);
      // Increase the size of send_buf if small
      if (nsendbyte > send_buf_max_byte_[n]) {
        send_buf_[n] = reinterpret_cast<byte *>(std::realloc(send_buf_[n],
                                                             nsendbyte));
        send_buf_max_byte_[n] = nsendbyte;
      }
      // Copy to send buffer
      std::memcpy(send_buf_[n],
                  reinterpret_cast<byte *>(exit_buf_(n).GetArray()), nsendbyte);
      exit_buf_(n).Reset();
      int err = MPI_Isend(send_buf_[n], nsendbyte, MPI_CHAR, ranklist_[n],
                          tag_rayt_, MPI_COMM_RAYT, &(send_req_[n]));
      if (DEBUG_RAYT_COMM) {
        if (err) {
          std::stringstream msg;
          msg << "### FATAL ERROR in PhotonCommunicator::RayTrace" << std::endl
              << "Rank " << Globals::my_rank
              << " MPI_Isend failed with err : " << err << std::endl;
          ATHENA_ERROR(msg);
        }
      }
      nsendproc_++;
    }
  }
  return;
}

//--------------------------------------------------------------------------------------
//! \fn PhotonCommunicator::ProbeAndReceive()
//! \brief Check incoming messages and receive photons from neighbor ranks.

template<typename T>
void PhotonCommunicator<T>::ProbeAndReceive() {
  MPI_Status status_probe;
  int flag_recv=1, nrecvbyte, nrecv;
  int err;
  while (flag_recv) {
    err = MPI_Iprobe(MPI_ANY_SOURCE, tag_rayt_,
                         MPI_COMM_RAYT, &flag_recv, &status_probe);
    if (DEBUG_RAYT_COMM) {
      if (err) {
        std::stringstream msg;
        msg << "### FATAL ERROR in PhotonCommunicator::RayTrace" << std::endl
            << "MPI_Iprobe failed with err : " << err << std::endl;
        ATHENA_ERROR(msg);
      }
    }
    if (flag_recv) { // There is incoming message
      err = MPI_Get_count(&status_probe, MPI_CHAR, &nrecvbyte);
      nrecv = nrecvbyte/sizeof(T);
      // Resize recv_buf if small
      if (nrecvbyte > recv_buf_max_byte_) {
        recv_buf_ = reinterpret_cast<byte *>(std::realloc(recv_buf_, nrecvbyte));
        recv_buf_max_byte_ = nrecvbyte;
      }
      // Blocking receive
      err = MPI_Recv(recv_buf_, nrecvbyte, MPI_CHAR, status_probe.MPI_SOURCE,
                     tag_rayt_, MPI_COMM_RAYT, MPI_STATUS_IGNORE);
      if (DEBUG_RAYT_COMM) {
        if (err) {
          std::stringstream msg;
          msg << "### FATAL ERROR in PhotonCommunicator::RayTrace" << std::endl
              << "MPI_Recv failed with err : " << err << std::endl;
          ATHENA_ERROR(msg);
        }
      }
      my_buf_.PushMultiple(recv_buf_, nrecv);
    }
  }
  return;
}


//--------------------------------------------------------------------------------------
//! \fn PhotonCommunicator::CompleteSendRequests()
//! \brief Complete enabled send operations.

template<typename T>
void PhotonCommunicator<T>::CompleteSendRequests() {
  if (nsendproc_ > 0) {
    int nsenddone = 0;
    int err = MPI_Testsome(nngbr_, send_req_, &nsenddone, nidsenddone_,
                           MPI_STATUSES_IGNORE);
    if (DEBUG_RAYT_COMM) {
      if (err) {
        std::stringstream msg;
        msg << "### FATAL ERROR in PhotonCommunicator::RayTrace" << std::endl
            << "Rank " << Globals::my_rank
            << " MPI_Testsome failed with err : " << err << std::endl;
        ATHENA_ERROR(msg);
      }
    }
    if (nsenddone > 0) {
      nsendproc_ -= nsenddone;
      for (int n=0; n<nsenddone; ++n) {
        send_req_[nidsenddone_[n]] = MPI_REQUEST_NULL;
      }
    }
  }
  return;
}


//--------------------------------------------------------------------------------------
//! \fn PhotonCommunicator::CheckAndUpdateDestroyCounter()
//! \brief Check and update destroy counter using remote memory access operations

template<typename T>
void PhotonCommunicator<T>::CheckAndUpdateDestroyCounter() {
  int err;
  if (Globals::my_rank != 0) {
    if (my_ndest_to_accum_ > 0ULL) {
      err = MPI_Fetch_and_op(&my_ndest_to_accum_, destcnt_,
                             MPI_UNSIGNED_LONG_LONG, 0, 0, MPI_SUM, win_);
      MPI_Win_flush_local(0, win_);
      my_ndest_to_accum_ = 0ULL;
    }
  }
  if (Globals::my_rank == 0) {
    if ((*destcnt_ + my_ndest_to_accum_) == ndest_max_) {
      all_done_ = 1;
      all_done_others_ = 1;
      err = MPI_Ibcast(&all_done_others_, 1, MPI_INT, 0,
                       MPI_COMM_RAYT, &all_done_req_);
      if (DEBUG_RAYT_COMM) {
        if (err) {
          std::stringstream msg;
          msg << "### FATAL ERROR in PhotonCommunicator::RayTrace" << std::endl
              << "Rank " << Globals::my_rank
              << " MPI_Ibcast failed with err : " << err << std::endl;
          ATHENA_ERROR(msg);
        }
      }
    }
  } else {
    MPI_Test(&all_done_req_, &flag_all_done_, MPI_STATUS_IGNORE);
    if (DEBUG_RAYT_COMM) {
      if (err) {
        std::stringstream msg;
        msg << "### FATAL ERROR in PhotonCommunicator::RayTrace" << std::endl
            << "Rank " << Globals::my_rank
            << " MPI_Test failed with err : " << err << std::endl;
        ATHENA_ERROR(msg);
      }
    }
    if (flag_all_done_)
      all_done_ = 1;
  }
  return;
}

//--------------------------------------------------------------------------------------
//! \fn PhotonCommunicator::CreateMPIWindow()
//! \brief Create an MPI window object for RMA operations

template<typename T>
void PhotonCommunicator<T>::CreateMPIWindow() {
  // Reference: window creation and ordering in MPI report
  MPI_Info_create(&win_info_);
  // "if set to true, then the implementation may assume that the argument size is
  // identical on all processes, and that all processes have provided this info key with
  // the same value"
  MPI_Info_set(win_info_, "same_size", "true");
  // "if set to true, then the implementation may assume that the argument disp_unit is
  // identical on all processes, and that all processes have provided this info key with
  // the same value"
  MPI_Info_set(win_info_, "same_disp_unit", "true");
  // "if set to same_op, the implementation will assume that all concurrent accumulate
  // calls to the same target address will use the same operation"
  MPI_Info_set(win_info_, "accumulate_ops", "same_op");
  // "The default strict ordering may incur a significant performance penalty...If set to
  // none, then no ordering will be guaranteed for accumulate calls
  MPI_Info_set(win_info_, "accumulate_ordering", "none");

  int err;
  err = MPI_Win_allocate((MPI_Aint)sizeof(std::uint64_t), sizeof(std::uint64_t),
                         win_info_, MPI_COMM_RAYT, &destcnt_, &win_);
  if (err) {
    std::stringstream msg;
    msg << "### FATAL ERROR in PhotonCommunicator::CreateMPIWindow" << std::endl
        << "Rank " << Globals::my_rank
        << " MPI_Win_allocate failed with err : " << err << std::endl;
    ATHENA_ERROR(msg);
  }
  err = MPI_Win_lock_all(MPI_MODE_NOCHECK, win_);
  if (err) {
    std::stringstream msg;
    msg << "### FATAL ERROR in PhotonCommunicator::CreateMPIWindow" << std::endl
        << "Rank " << Globals::my_rank
        << " MPI_Win_lock_all failed with err : " << err << std::endl;
    ATHENA_ERROR(msg);
  }
  return;
}


//--------------------------------------------------------------------------------------
//! \fn PhotonCommunicator::DestroyMPIWindow()
//! \brief Destroy an MPI window object

template<typename T>
void PhotonCommunicator<T>::DestroyMPIWindow() {
  MPI_Win_unlock_all(win_);
  MPI_Info_free(&win_info_);
  int err = MPI_Win_free(&win_);
  if (err) {
    std::stringstream msg;
    msg << "### FATAL ERROR in PhotonCommunicator::DestroyMPIWindow " << std::endl
        << "Rank " << Globals::my_rank
        << " MPI_Win_free failed with err : " << err << std::endl;
    ATHENA_ERROR(msg);
  }
  return;
}

#endif // MPI_PARALLEL

template class PhotonCommunicator<Photon>;
