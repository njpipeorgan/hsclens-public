#!/bin/bash
#----------------------------------------------------

#SBATCH -J rockstar
#SBATCH -o job.o%j
#SBATCH -e job.e%j
#SBATCH -p _QUEUE_
#SBATCH -N 1
#SBATCH -n _NUM_THREADS_
#SBATCH -t 10:00:00

cd ./halos/ &&
bash ./run-rockstar.sh &&
cd ..

