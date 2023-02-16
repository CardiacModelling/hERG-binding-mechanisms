#!/bin/bash
#SBATCH --partition             serial-normal
#SBATCH --ntasks                1
#SBATCH --nodes                 1
#SBATCH --tasks-per-node        1
#SBATCH --cpus-per-task         4
#SBATCH --array                 1-8
#SBATCH --time                  72:00:00
#SBATCH --mem                   62G
#SBATCH --job-name              compare-qnet
#SBATCH --output                log/compare-qnet.%A_%a.out
#SBATCH --error                 log/compare-qnet.%A_%a.err
##SBATCH --mail-type             ALL
##SBATCH --mail-user            chonloklei@um.edu.mo

source /etc/profile
source /etc/profile.d/modules.sh
source /home/chonloklei/m  # Load miniconda

ulimit -s unlimited

# Load module
module purge

# Path and Python version checks
pwd
python --version
conda activate /home/chonloklei/hERG-binding/env  # Load miniconda venv
python --version
which python

# Set up

# Run
echo ${SLURM_ARRAY_TASK_ID}
python -u compare-qnets.py -s -m ${SLURM_ARRAY_TASK_ID}
#python -u compare-qnets.py -s -v -m ${SLURM_ARRAY_TASK_ID}

echo "Done."
