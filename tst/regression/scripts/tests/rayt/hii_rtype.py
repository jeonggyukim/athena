# Regression test based on expansion of an R-type ionization front
#
# Test of simple photochemistry. Hydrodynamics is turned off. See Sec 3.1.1 in KimJG et
# al. (2017).

# Modules
import logging
import os
import numpy as np
import sys
import scripts.utils.athena as athena
sys.path.insert(0, '../../vis/python')
import athena_read  # noqa
athena_read.check_nan_flag = True
logger = logging.getLogger('athena' + __name__[7:])  # set logger name based on module

nproc = 4            # Number of processors for MPI run
ncell = 64           # Number of cells
Qi = 1e49            # Ionizing photon rate of the central source [1/s]
tgas_HII = 8000      # Temperature of fully ionized gas [K]
tgas_HI = 100        # Temperature of fully neutral gas [K]
n0 = 1e2             # Hydrogen number density of background medium [cm^-3]
tol_rel = 1e-1       # Maximum relative tolerance for ionization front radius
tol_rel_tlim = 1e-2  # Relative tolerance for ionization front radius at tlim


# Prepare Athena++
def prepare(**kwargs):
    logger.debug('Running test ' + __name__)

    # Compile with MPI
    athena.configure('mpi',
                     prob='hii',
                     nscalars=1,
                     nfreq_rayt=1,
                     **kwargs)
    athena.make()
    os.system('mv bin/athena bin/athena_mpi_hii')
    os.system('mv obj obj_mpi_hii')

    return None


def run(**kwargs):
    arguments = ['job/problem_id=hii_rtype',
                 'output2/dt=-1',
                 'output3/dt=-1',
                 'mesh/nx1=' + repr(ncell),
                 'problem/Qi=' + repr(Qi),
                 'photchem/mode=simple',
                 'photchem/f_dt_rad=0.1',
                 'photchem/tgas_HII=' + repr(tgas_HII),
                 'photchem/tgas_HI=' + repr(tgas_HI)
                 ]

    os.system('mv obj_mpi_hii obj')
    os.system('mv bin/athena_mpi_hii bin/athena')
    athena.mpirun(kwargs['mpirun_cmd'], kwargs['mpirun_opts'],
                  nproc, 'ray_tracing/athinput.hii_rtype', arguments)

    return 'skip_lcov'


def analyze():
    analyze_status = True

    fname = os.path.join('bin', 'hii_rtype.hst')
    h = athena_read.hst(fname)
    # Drop time=0 data
    for k, v in h.items():
        h[k] = h[k][1:]

    pc_cgs = 3.08567758e+18
    kms_cgs = 1e5
    tunit = (pc_cgs/kms_cgs)

    # Analytic solution using the default input parameters
    def calc_radius_ifront(t, Qi=1e49, n=n0, Tion=tgas_HII):
        alphaB = 2.56e-13*(Tion*1e-4)**-0.8
        trec = 1/(alphaB*n)
        R0_pc = (3.0*Qi/(4.0*np.pi*alphaB*n**2))**(1/3)/pc_cgs
        return R0_pc*(1 - np.exp(-t/trec))**(1/3)

    sol = calc_radius_ifront(h['time']*tunit)
    sim = h['IF_vol_r']/h['IF_vol']
    err = (sim - sol)/sol
    logger.info('Maximum/minimum relative error in ionization front radius: %g %g',
                max(err), min(err))
    logger.info('Relative difference in ionization front radius at tlim: %g', err[-1])

    if abs(max(err)) > tol_rel or abs(min(err)) > tol_rel or abs(err[-1]) > tol_rel_tlim:
        analyze_status = False

    return analyze_status
