# Regression test based on radiation in vacuum
#
# Performs adaptive ray tracing by placing a point source at the center and setting
# opacity to a very small value. Checks errors for r^2*rad_energy_density = lum/(4*pi*c).

# Modules
import logging
import os
import numpy as np
import sys
import copy
import glob
import scripts.utils.athena as athena
sys.path.insert(0, '../../vis/python')
import athena_read                             # noqa
athena_read.check_nan_flag = True
logger = logging.getLogger('athena' + __name__[7:])

ncell = 128
ncell_mb = 64
rays_per_cell = [2.0, 4.0, 8.0]
nproc = 4  # for MPI runs
tolerance = [0.095, 0.041, 0.025]  # Tolerance for mean error (no rotation)
tolerance_rel = 1e-5  # Relative tolerance for serial and MPI runs


# Prepare Athena++
def prepare(**kwargs):
    logger.debug('Running test ' + __name__)

    # Compile with MPI
    athena.configure('mpi',
                     prob='rayt',
                     **kwargs)
    athena.make()
    os.system('mv bin/athena bin/athena_mpi_rayt')
    os.system('mv obj obj_mpi_rayt')

    # Compile without MPI
    athena.configure(prob='rayt',
                     **kwargs)
    athena.make()
    os.system('mv bin/athena bin/athena_rayt')
    os.system('mv obj obj_rayt')

    return None


def run(**kwargs):
    arguments_tmp = ['job/problem_id=rayt_{0:s}_rpc{1:04d}',
                     'ray_tracing/rays_per_cell={0:f}',
                     'ray_tracing/rotate_rays=false',
                     'time/nlim=1',
                     'mesh/nx1=' + repr(ncell),
                     'mesh/x1min=' + repr(-ncell/2),
                     'mesh/x1max=' + repr(ncell/2),
                     'mesh/nx2=' + repr(ncell),
                     'mesh/x2min=' + repr(-ncell/2),
                     'mesh/x2max=' + repr(ncell/2),
                     'mesh/nx3=' + repr(ncell),
                     'mesh/x3min=' + repr(-ncell/2),
                     'mesh/x3max=' + repr(ncell/2),
                     'meshblock/nx1=' + repr(ncell_mb),
                     'meshblock/nx2=' + repr(ncell_mb),
                     'meshblock/nx3=' + repr(ncell_mb)]

    # Test with MPI
    os.system('mv obj_mpi_rayt obj')
    os.system('mv bin/athena_mpi_rayt bin/athena')
    for rpc in rays_per_cell:
        arguments = copy.deepcopy(arguments_tmp)
        arguments[0] = arguments[0].format('mpi', int(rpc))
        arguments[1] = arguments[1].format(rpc)
        athena.mpirun(kwargs['mpirun_cmd'], kwargs['mpirun_opts'],
                      nproc, 'ray_tracing/athinput.rayt', arguments)

    # Test without MPI
    os.system('mv obj_rayt obj')
    os.system('mv bin/athena_rayt bin/athena')
    for rpc in rays_per_cell:
        arguments = copy.deepcopy(arguments_tmp)
        arguments[0] = arguments[0].format('serial', int(rpc))
        arguments[1] = arguments[1].format(rpc)
        athena.run('ray_tracing/athinput.rayt', arguments)

    return 'skip_lcov'


def analyze():
    analyze_status = True

    def read_vacuum(problem_id, num=0):
        nmb = len(glob.glob(os.path.join('bin', problem_id
                                         + '.block*.out1.{0:05d}.vtk'.format(num))))
        bid = range(0, nmb)
        r = np.array([])
        rsq_erad = np.array([])
        for i, bid_ in enumerate(bid):
            fname = os.path.join('bin', problem_id
                                 + '.block{0:d}.out1.{1:05d}.vtk'.format(bid_, num))
            x1f, x2f, x3f, dat = athena_read.vtk(fname)
            x1 = 0.5*(x1f[1:] + x1f[:-1])
            x2 = 0.5*(x2f[1:] + x2f[:-1])
            x3 = 0.5*(x3f[1:] + x3f[:-1])
            z, y, x = np.meshgrid(x3, x2, x1, indexing='ij')
            rsq = x**2 + y**2 + z**2
            r = np.append(r, (np.sqrt(rsq)).flatten())
            rsq_erad = np.append(rsq_erad, (rsq*dat['Er_rayt0']).flatten())

        return r, rsq_erad

    for rpc, tol in zip(rays_per_cell, tolerance):
        # Compute mean and median absolute errors of 4pi*r^2*Erad
        problem_id = 'rayt_mpi_rpc{0:04d}'.format(int(rpc))
        r, rsq_erad = read_vacuum(problem_id)
        err_mean_mpi = np.mean(np.abs((rsq_erad*4.0*np.pi - 1.0)))
        err_med_mpi = np.median(np.abs((rsq_erad*4.0*np.pi - 1.0)))

        problem_id = 'rayt_serial_rpc{0:04d}'.format(int(rpc))
        r, rsq_erad = read_vacuum(problem_id)
        err_mean_serial = np.mean(np.abs((rsq_erad*4.0*np.pi - 1.0)))
        err_med_serial = np.median(np.abs((rsq_erad*4.0*np.pi - 1.0)))

        logger.info('Rays per cell, Mean, median errors: %d %g %g (serial)',
                    rpc, err_mean_serial, err_med_serial)
        logger.info('Rays per cell, Mean, median errors: %d %g %g (mpi)',
                    rpc, err_mean_mpi, err_med_mpi)

        if err_mean_serial > tol or err_mean_mpi > tol:
            analyze_status = False

        if np.abs((err_mean_serial - err_mean_mpi)
                  / (err_mean_serial + err_mean_mpi)) > 1e-5:
            analyze_status = False

    return analyze_status
