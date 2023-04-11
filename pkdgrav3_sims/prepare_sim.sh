# Omega_M, Omega_Lambda, sigma_8 are read from bash arguments
OMEGA_M=$1
OMEGA_L=$2
SIGMA_8=$3

# Queue name and number of threads are also read from bash arguments
QUEUE=${4:-skx-normal}
NUM_THREADS=${5:-48}

# Set path to pkdgrav3 and MUSIC here
PKDGRAV3_PATH=$WORK2/pkdgrav3/build/pkdgrav3
MUSIC_PATH=$WORK2/music/MUSIC

# Planck values
OMEGA_B=0.0493
N_SPEC=0.9649
HUBBLE_CONSTANT=67.36
HUBBLE_PARAMETER=0.6736
PARTICLE_MASS=$(python -c "print(2.7753312e11*((500.0/1024.0)**3)*${OMEGA_M})")
# ((3 H^2)/(8 \pi G)) == 2.7753312e11 (Msun/h)/(Mpc/h)^3

SIM_NAME="sim_Om${OMEGA_M}_si${SIGMA_8}_1024_500"
SCRIPT_DIRECTORY=$(dirname "$0")
WORKING_DIRECTORY=$(pwd)/${SIM_NAME}
echo NUM_THREADS = ${NUM_THREADS}
echo SCRIPT_DIRECTORY = ${SCRIPT_DIRECTORY}
echo WORKING_DIRECTORY = ${WORKING_DIRECTORY}

if test -f ./${SIM_NAME}; then
  echo ${SIM_NAME} already exists.
  exit 0
fi

cp -r ${SCRIPT_DIRECTORY}/template_Om0.0000_si0.0000_1024_500 ${WORKING_DIRECTORY}

FILES="cosmology.par job.slurm halo.slurm ics/parameters.conf halos/parallel.cfg"
for f in $FILES; do
  echo replacing file ${WORKING_DIRECTORY}/${f}
  sed -i "s#_WORKING_DIRECTORY_#${WORKING_DIRECTORY}#" ${WORKING_DIRECTORY}/${f}
  sed -i "s/_QUEUE_/${QUEUE}/"                         ${WORKING_DIRECTORY}/${f}
  sed -i "s/_NUM_THREADS_/${NUM_THREADS}/"             ${WORKING_DIRECTORY}/${f}
  sed -i "s#_PKDGRAV3_PATH_#${PKDGRAV3_PATH}#"         ${WORKING_DIRECTORY}/${f}
  sed -i "s#_MUSIC_PATH_#${MUSIC_PATH}#"               ${WORKING_DIRECTORY}/${f}

  sed -i "s/_OMEGA_M_/${OMEGA_M}/"                     ${WORKING_DIRECTORY}/${f}
  sed -i "s/_OMEGA_L_/${OMEGA_L}/"                     ${WORKING_DIRECTORY}/${f}
  sed -i "s/_SIGMA_8_/${SIGMA_8}/"                     ${WORKING_DIRECTORY}/${f}
  sed -i "s/_OMEGA_B_/${OMEGA_B}/"                     ${WORKING_DIRECTORY}/${f}
  sed -i "s/_N_SPEC_/${N_SPEC}/"                       ${WORKING_DIRECTORY}/${f}
  sed -i "s/_HUBBLE_CONSTANT_/${HUBBLE_CONSTANT}/"     ${WORKING_DIRECTORY}/${f}
  sed -i "s/_HUBBLE_PARAMETER_/${HUBBLE_PARAMETER}/"   ${WORKING_DIRECTORY}/${f}
  sed -i "s/_PARTICLE_MASS_/${PARTICLE_MASS}/"         ${WORKING_DIRECTORY}/${f}
done
