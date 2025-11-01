#!/bin/bash

# This is a setup for GPU training on Chicoma. Find out how much
# memory per node, number of CPUs/node.

# NOTE: Number of CPUs per GPU must be an even number since there are
# 2 threads per core. If an odd number is requested the next higher
# even number gets used. 

# The following are one set of SBATCH options for the Chicoma GPU
# partition. There are optional other constraints. Namely, `gpu80` and
# `gpu40only`. 

#SBATCH --job-name=loderunner_study<studyIDX>
#SBATCH --account=w25_artimis_g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
##SBATCH --partition=gpu_debug
##SBATCH --reservation=gpu_debug
##SBATCH --time=2:00:00
#SBATCH --partition=gpu
#SBATCH --time=16:00:00
#SBATCH --output=study<studyIDX>_epoch0001.out
#SBATCH --error=study<studyIDX>_epoch0001.err
#SBATCH -vvv


# Check available GPUs
sinfo  -o "%P %.24G %N"
srun /usr/bin/echo $CUDA_AVAILABLE_DEVICES

# Load correct conda environment
module load python/3.11-anaconda-2023.07
source activate
conda activate torch_ch_gpu_241112

# Currently need to set an environment variable for MKL
# threading. Believed to be related to Numpy.
export MKL_SERVICE_FORCE_INTEL=TRUE

# Get start time
export date00=`date`

# Start the Code
srun python <train_script> @study<studyIDX>_START.input

# Get end time and print to stdout
export date01=`date`

echo "===================TIME STARTED==================="
echo $date00
echo "===================TIME FINISHED==================="
echo $date01
