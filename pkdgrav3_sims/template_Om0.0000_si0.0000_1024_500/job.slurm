#!/bin/bash
#----------------------------------------------------

#SBATCH -J nbody
#SBATCH -o job.o%j
#SBATCH -e job.e%j
#SBATCH -p _QUEUE_
#SBATCH -N 1
#SBATCH -n _NUM_THREADS_
#SBATCH -t 48:00:00

cd _WORKING_DIRECTORY_

ml gcc fftw3 gsl hdf5
cd ics
time OMP_NUM_THREADS=_NUM_THREADS_ _MUSIC_PATH_ parameters.conf
cd ..

ml intel
time _PKDGRAV3_PATH_ -sz _NUM_THREADS_ cosmology.par
