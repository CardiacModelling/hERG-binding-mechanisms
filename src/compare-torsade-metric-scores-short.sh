#!/bin/bash
#SBATCH --partition             serial-short
#SBATCH --ntasks                1
#SBATCH --nodes                 1
#SBATCH --tasks-per-node        1
#SBATCH --cpus-per-task         4
#SBATCH --array                 1-4
#SBATCH --time                  24:00:00
#SBATCH --mem                   62G
#SBATCH --job-name              compare-tms-v
#SBATCH --output                log/compare-tms-v.%A_%a.out
#SBATCH --error                 log/compare-tms-v.%A_%a.err
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
echo $((14-${SLURM_ARRAY_TASK_ID}))
python -u compare-torsade-metric-scores.py -s -m $((14-${SLURM_ARRAY_TASK_ID}))
#python -u compare-torsade-metric-scores.py -s -v -m $((14-${SLURM_ARRAY_TASK_ID}))

echo "Done."
