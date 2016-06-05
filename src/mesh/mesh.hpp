#ifndef MESH_HPP
#define MESH_HPP
//======================================================================================
// Athena++ astrophysical MHD code
// Copyright (C) 2014 James M. Stone  <jmstone@princeton.edu>
// See LICENSE file for full public license information.
//======================================================================================
//! \file mesh.hpp
//  \brief defines Mesh and MeshBlock classes, and various structs used in them
//  The Mesh is the overall grid structure, and MeshBlocks are local patches of data
//  (potentially on different levels) that tile the entire domain.
//======================================================================================

// C/C++ headers
#include <stdint.h>  // int64_t

// Athena++ classes headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "meshblock_tree.hpp"
#include "../outputs/wrapper.hpp"
#include "../task_list/task_list.hpp"
#include "../bvals/bvals.hpp"
#include "mesh_refinement.hpp"

// Forward declarations
class ParameterInput;
class Mesh;
class MeshRefinement;
class MeshBlockTree;
class BoundaryValues;
class TaskList;
class Coordinates;
class Reconstruction;
class Hydro;
class Field;
class EquationOfState;

//--------------------------------------------------------------------------------------
//! \struct NeighborBlock
//  \brief neighbor rank, level, and ids

typedef struct NeighborBlock {
  int rank, level, gid, lid, ox1, ox2, ox3, fi1, fi2, bufid, eid, targetid;
  enum NeighborType type;
  enum BoundaryFace fid;
  bool polar; // flag indicating boundary is across a pole

  NeighborBlock() : rank(-1), level(-1), gid(-1), lid(-1), ox1(-1), ox2(-1), ox3(-1),
    bufid(-1), targetid(-1), fi1(-1), fi2(-1), eid(-1), type(NEIGHBOR_NONE),
    fid(FACE_UNDEF), polar(false) {};

  void SetNeighbor(int irank, int ilevel, int igid, int ilid, int iox1, int iox2,
                   int iox3, enum NeighborType itype, int ibid, int itargetid,
                   bool ipolar, int ifi1, int ifi2);
} NeighborBlock;

//--------------------------------------------------------------------------------------
//! \struct PolarNeighborBlock
//  \brief Struct for describing neighbors around pole at same radius and polar angle

typedef struct PolarNeighborBlock {
  int rank;    // MPI rank of neighbor
  int lid;     // local ID of neighbor
  int gid;     // global ID of neighbor
  bool north;  // flag that is true for North pole and false for South pole
} PolarNeighborBlock;

//--------------------------------------------------------------------------------------
//! \struct RegionSize
//  \brief physical size and number of cells in a Mesh

typedef struct RegionSize {
  Real x1min, x2min, x3min;
  Real x1max, x2max, x3max;
  Real x1rat, x2rat, x3rat; // ratio of x(i)/x(i-1)
  int nx1, nx2, nx3;        // number of active cells (not including ghost zones)
} RegionSize;

//--------------------------------------------------------------------------------------
//! \class MeshBlock
//  \brief data/functions associated with a single block

class MeshBlock {
  friend class RestartOutput;
  friend class BoundaryValues;
  friend class Mesh;
  friend class Hydro;
  friend class TaskList;
#ifdef HDF5OUTPUT
  friend class ATHDF5Output;
#endif

public:
  MeshBlock(int igid, int ilid, LogicalLocation iloc, RegionSize input_size,
    enum BoundaryFlag *input_bcs, Mesh *pm, ParameterInput *pin, bool ref_flag = false);
  MeshBlock(int igid, int ilid, Mesh *pm, ParameterInput *pin, LogicalLocation iloc,
    RegionSize input_block, enum BoundaryFlag *input_bcs, Real icost, char *mbdata);
  ~MeshBlock();

  //data
  Mesh *pmy_mesh;  // ptr to Mesh containing this MeshBlock
  LogicalLocation loc;
  RegionSize block_size;
  enum BoundaryFlag block_bcs[6];
  int nblevel[3][3][3];
  int is,ie,js,je,ks,ke;
  int gid, lid;
  int cis,cie,cjs,cje,cks,cke,cnghost;

  // user MeshBlock data that can be stored in restart files
  AthenaArray<Real> *rusermeshblockdata;
  AthenaArray<int> *iusermeshblockdata;

  // mesh-related objects
  Coordinates *pcoord;
  BoundaryValues *pbval;
  Reconstruction *precon;
  MeshRefinement *pmr;

  // physics-related objects
  Hydro *phydro;
  Field *pfield;
  EquationOfState *peos;

  MeshBlock *prev, *next;

  // functions
  size_t GetBlockSizeInBytes(void);
  void SearchAndSetNeighbors(MeshBlockTree &tree, int *ranklist, int *nslist);
  void UserWorkInLoop(void); // in ../pgen
  void InitUserMeshBlockData(ParameterInput *pin); // in ../pgen

private:
  // data
  NeighborBlock neighbor[56];
  PolarNeighborBlock *polar_neighbor_north, *polar_neighbor_south;
  int nneighbor;
  Real cost;
  Real new_block_dt;
  unsigned long int finished_tasks[4];
  int indx_first_task_, num_tasks_left_;
  int nreal_user_meshblock_data_, nint_user_meshblock_data_;

  // functions
  void AllocateRealUserMeshBlockDataField(int n);
  void AllocateIntUserMeshBlockDataField(int n);
  void ProblemGenerator(ParameterInput *pin); // in ../pgen
};

//--------------------------------------------------------------------------------------
//! \class Mesh
//  \brief data/functions associated with the overall mesh

class Mesh {
  friend class RestartOutput;
  friend class MeshBlock;
  friend class BoundaryValues;
  friend class Coordinates;
  friend class MeshRefinement;
  friend class HydroSourceTerms;
#ifdef HDF5OUTPUT
  friend class ATHDF5Output;
#endif

public:
  Mesh(ParameterInput *pin, int test_flag=0);
  Mesh(ParameterInput *pin, IOWrapper &resfile, int test_flag=0);
  ~Mesh();

  // data
  RegionSize mesh_size;
  enum BoundaryFlag mesh_bcs[6];
  Real start_time, tlim, cfl_number, time, dt;
  int nlim, ncycle;
  bool adaptive, multilevel, face_only;

  TaskList *ptlist;
  MeshBlock *pblock;

  AthenaArray<Real> *rusermeshdata;
  AthenaArray<int> *iusermeshdata;

  // functions
  int64_t GetTotalCells(void);
  int GetNumMeshThreads() const {return num_mesh_threads_;}
  void Initialize(int res_flag, ParameterInput *pin);
  void SetBlockSizeAndBoundaries(LogicalLocation loc, RegionSize &block_size,
                                 enum BoundaryFlag *block_bcs);
  void CompleteAllMeshTaskLists(void);
  void NewTimeStep(void);
  void AdaptiveMeshRefinement(ParameterInput *pin);
  unsigned int CreateAMRMPITag(int lid, int ox1, int ox2, int ox3);
  MeshBlock* FindMeshBlock(int tgid);
  void UserWorkAfterLoop(ParameterInput *pin); // method in ../pgen

private:
  // data
  int root_level, max_level, current_level;
  int nbtotal;
  int maxneighbor_;
  int num_mesh_threads_;
  int *nslist, *ranklist, *nblist;
  Real *costlist;
  int *nref, *nderef, *bnref, *bnderef, *rdisp, *brdisp, *ddisp, *bddisp;
  LogicalLocation *loclist;
  MeshBlockTree tree;
  long int nrbx1, nrbx2, nrbx3;
  bool user_meshgen_[3];
  int nreal_user_mesh_data_, nint_user_mesh_data_;

  // functions
  MeshGenFunc_t MeshGenerator_[3];
  SrcTermFunc_t UserSourceTerm_;
  BValFunc_t BoundaryFunction_[6];
  AMRFlagFunc_t AMRFlag_;
  void AllocateRealUserMeshDataField(int n);
  void AllocateIntUserMeshDataField(int n);
  void OutputMeshStructure(int dim);
  void LoadBalance(Real *clist, int *rlist, int *slist, int *nlist, int nb);

  // methods in ../pgen
  void InitUserMeshData(ParameterInput *pin);
  void EnrollUserBoundaryFunction (enum BoundaryFace face, BValFunc_t my_func);
  void EnrollUserRefinementCondition(AMRFlagFunc_t amrflag);
  void EnrollUserMeshGenerator(enum CoordinateDirection dir, MeshGenFunc_t my_mg);
  void EnrollUserSourceTermFunction(SrcTermFunc_t my_func);
};

//--------------------------------------------------------------------------------------
// \!fn Real DefaultMeshGeneratorX1(Real x, RegionSize rs)
// \brief x1 mesh generator function, x is the logical location; x=i/nx1

inline Real DefaultMeshGeneratorX1(Real x, RegionSize rs)
{
  Real lw, rw;
  if(rs.x1rat==1.0) {
    rw=x, lw=1.0-x;
  } else {
    Real ratn=pow(rs.x1rat,rs.nx1);
    Real rnx=pow(rs.x1rat,x*rs.nx1);
    lw=(rnx-ratn)/(1.0-ratn);
    rw=1.0-lw;
  }
  return rs.x1min*lw+rs.x1max*rw;
}

//--------------------------------------------------------------------------------------
// \!fn Real DefaultMeshGeneratorX2(Real x, RegionSize rs)
// \brief x2 mesh generator function, x is the logical location; x=j/nx2

inline Real DefaultMeshGeneratorX2(Real x, RegionSize rs)
{
  Real lw, rw;
  if(rs.x2rat==1.0) {
    rw=x, lw=1.0-x;
  } else {
    Real ratn=pow(rs.x2rat,rs.nx2);
    Real rnx=pow(rs.x2rat,x*rs.nx2);
    lw=(rnx-ratn)/(1.0-ratn);
    rw=1.0-lw;
  }
  return rs.x2min*lw+rs.x2max*rw;
}

//--------------------------------------------------------------------------------------
// \!fn Real DefaultMeshGeneratorX3(Real x, RegionSize rs)
// \brief x3 mesh generator function, x is the logical location; x=k/nx3

inline Real DefaultMeshGeneratorX3(Real x, RegionSize rs)
{
  Real lw, rw;
  if(rs.x3rat==1.0) {
    rw=x, lw=1.0-x;
  } else {
    Real ratn=pow(rs.x3rat,rs.nx3);
    Real rnx=pow(rs.x3rat,x*rs.nx3);
    lw=(rnx-ratn)/(1.0-ratn);
    rw=1.0-lw;
  }
  return rs.x3min*lw+rs.x3max*rw;
}

#endif  // MESH_HPP
